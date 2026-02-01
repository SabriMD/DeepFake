import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List
import sys
from tqdm import tqdm

from module2_vlm_reasoner import VLMLogicReasoner

from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torch.nn import functional as F

class ForensicSignalDetectorInference(nn.Module):
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        
        print("LOADING MODULE 1: FORENSIC SIGNAL DETECTOR")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {device}")
        
        # Load the trained model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Image preprocessing (matching training)
        self.transform = Compose([
            Resize((182, 182)),
            ToTensor(),
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
        
        print(f"Module 1 loaded successfully")
    
    def _load_model(self, checkpoint_path: str):
        """Load the trained PyTorch Lightning model"""
        
        # Import the model class from your training script
        # We'll reconstruct it here
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', 
            'efficient_x3d_xs', 
            pretrained=False  # We'll load our trained weights
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model state dict (PyTorch Lightning format)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Remove 'video_model.' prefix and load into our model
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('video_model.'):
                    new_key = key.replace('video_model.', '')
                    model_state_dict[new_key] = value
            
            model.load_state_dict(model_state_dict, strict=False)
        
        # Add classification head (from your training code)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(400, 1)
        
        # Load head weights
        if 'state_dict' in checkpoint:
            linear_weight = checkpoint['state_dict'].get('linear.weight')
            linear_bias = checkpoint['state_dict'].get('linear.bias')
            if linear_weight is not None:
                self.linear.weight.data = linear_weight
                self.linear.bias.data = linear_bias
        
        return model.to(self.device)
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        img_tensor = self.transform(image)  # (3, 182, 182)
        
        # Replicate to create "video" with 13 identical frames
        # This simulates temporal consistency for the video model
        video_tensor = img_tensor.unsqueeze(1).repeat(1, 13, 1, 1)  # (3, 13, 182, 182)
        
        # Add batch dimension
        video_tensor = video_tensor.unsqueeze(0)  # (1, 3, 13, 182, 182)
        
        return video_tensor
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, float]:
        # Preprocess
        video_tensor = self.preprocess_image(image).to(self.device)
        
        # Forward pass through X3D backbone
        features = self.model(video_tensor)
        
        # Classification head
        features = self.relu(features)
        features = self.dropout(features)
        logits = self.linear(features)
        
        # Get probability
        forensic_score = torch.sigmoid(logits).item()
        
        # Confidence based on distance from decision boundary (0.5)
        confidence = abs(forensic_score - 0.5) * 2
        
        return {
            "forensic_score": round(forensic_score, 4),
            "confidence": round(confidence, 4)
        }

class DualModuleFusion:
    
    def __init__(self, 
                 forensic_weight: float = 0.6,
                 semantic_weight: float = 0.4):
        self.forensic_weight = forensic_weight
        self.semantic_weight = semantic_weight
        
        # Normalize weights
        total = forensic_weight + semantic_weight
        self.forensic_weight /= total
        self.semantic_weight /= total
        
        print(f"\n{'='*70}")
        print("FUSION STRATEGY INITIALIZED")
        print(f"Forensic Weight: {self.forensic_weight:.2%}")
        print(f"Semantic Weight: {self.semantic_weight:.2%}")
        print(f"Method: Confidence-weighted ensemble")
    
    def fuse_predictions(self, 
                        forensic_result: Dict,
                        semantic_result: Dict) -> Dict[str, any]:
        
        forensic_score = forensic_result["forensic_score"]
        forensic_conf = forensic_result["confidence"]
        
        semantic_score = semantic_result["semantic_score"]
        semantic_conf = semantic_result["confidence"]
        
        # Dynamic weight adjustment based on confidence
        # If one module is more confident, give it more weight
        conf_adjusted_forensic = self.forensic_weight * (1 + forensic_conf * 0.3)
        conf_adjusted_semantic = self.semantic_weight * (1 + semantic_conf * 0.3)
        
        # Normalize adjusted weights
        total_weight = conf_adjusted_forensic + conf_adjusted_semantic
        conf_adjusted_forensic /= total_weight
        conf_adjusted_semantic /= total_weight
        
        # Calculate final authenticity score
        final_score = (
            conf_adjusted_forensic * forensic_score +
            conf_adjusted_semantic * semantic_score
        )
        
        # Determine manipulation type (prefer VLM's classification)
        manipulation_type = semantic_result["manipulation_type"]
        
        # If forensic is very confident but VLM says authentic, reconsider
        if forensic_score > 0.8 and forensic_conf > 0.7:
            if semantic_score < 0.3:
                manipulation_type = "subtle_manipulation"
        
        # Use VLM reasoning
        vlm_reasoning = semantic_result["vlm_reasoning"]
        
        return {
            "authenticity_score": round(final_score, 4),
            "manipulation_type": manipulation_type,
            "vlm_reasoning": vlm_reasoning,
            "forensic_score": round(forensic_score, 4),
            "semantic_score": round(semantic_score, 4),
            "forensic_confidence": round(forensic_conf, 4),
            "semantic_confidence": round(semantic_conf, 4)
        }

class DeepfakeDetectionPipeline:
    
    def __init__(self,
                 forensic_checkpoint: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_lightweight_vlm: bool = True):
        
        self.device = device
        print("INITIALIZING DEEPFAKE DETECTION PIPELINE")
        # Load Module 1: Forensic Detector
        self.forensic_detector = ForensicSignalDetectorInference(
            checkpoint_path=forensic_checkpoint,
            device=device
        )
        
        # Load Module 2: VLM Reasoner
        self.vlm_reasoner = VLMLogicReasoner(
            use_lightweight=use_lightweight_vlm,
            device=device
        )
        
        # Initialize fusion strategy
        self.fusion = DualModuleFusion(
            forensic_weight=0.6,  # Forensic slightly more reliable
            semantic_weight=0.4
        )
        
        print(f"{'='*70}")
        print("✓ PIPELINE READY")
        print(f"{'='*70}\n")
    
    def predict_single_image(self, image_path: str) -> Dict[str, any]:
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Module 1: Forensic analysis
        forensic_result = self.forensic_detector.predict(image)
        
        # Module 2: VLM semantic analysis
        semantic_result = self.vlm_reasoner.analyze_image(image)
        
        # Fuse predictions
        final_result = self.fusion.fuse_predictions(forensic_result, semantic_result)
        
        # Add image name
        final_result["image_name"] = Path(image_path).name
        
        return final_result
    
    def predict_batch(self, 
                     input_dir: str,
                     output_file: str = "predictions.json",
                     image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']):
        
        input_path = Path(input_dir)
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        image_files = sorted(image_files)
        
        print(f"PROCESSING {len(image_files)} IMAGES")
        print(f"Input Directory: {input_dir}")
        print(f"Output File: {output_file}")
        
        # Process each image
        predictions = []
        
        for image_path in tqdm(image_files, desc="Analyzing images"):
            try:
                result = self.predict_single_image(str(image_path))
                
                # Format for competition (only required fields)
                prediction = {
                    "image_name": result["image_name"],
                    "authenticity_score": result["authenticity_score"],
                    "manipulation_type": result["manipulation_type"],
                    "vlm_reasoning": result["vlm_reasoning"]
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                print(f"\n⚠ Error processing {image_path.name}: {e}")
                # Add fallback prediction
                predictions.append({
                    "image_name": image_path.name,
                    "authenticity_score": 0.5,
                    "manipulation_type": "error",
                    "vlm_reasoning": f"Error during processing: {str(e)}"
                })
        
        # Save predictions
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Processed {len(predictions)} images")
        print(f"Predictions saved to: {output_file}")
        
        return predictions

def main():
    
    parser = argparse.ArgumentParser(
        description="Deepfake Detection System - Competition Submission"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing test images"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.json",
        help="Output JSON file path (default: predictions.json)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="module1_checkpoints/module1-forensic-epoch=01-val_f1=0.9314.ckpt",
        help="Path to Module 1 checkpoint"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight VLM for faster inference"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = DeepfakeDetectionPipeline(
        forensic_checkpoint=args.checkpoint,
        device=args.device,
        use_lightweight_vlm=args.lightweight
    )
    
    # Run inference
    predictions = pipeline.predict_batch(
        input_dir=args.input_dir,
        output_file=args.output_file
    )
    
    print("SUCCESS! Predictions ready for submission.")
    

if __name__ == "__main__":
    main()
