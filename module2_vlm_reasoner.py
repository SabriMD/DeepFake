import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Optional
import json
import os

# VLM Options (choose based on what's available)
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    VLM_AVAILABLE = "BLIP"
except ImportError:
    VLM_AVAILABLE = None

try:
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    VLM_AVAILABLE = "InstructBLIP"
except ImportError:
    pass


class VLMLogicReasoner:
    
    def __init__(self, 
                 vlm_model_name: str = "Salesforce/instructblip-vicuna-7b",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_lightweight: bool = True):
        self.device = device
        self.use_lightweight = use_lightweight
        
        print(f"\n{'='*70}")
        print("INITIALIZING MODULE 2: VLM LOGIC REASONER")
        print(f"{'='*70}")
        print(f"Device: {device}")
        
        # Load VLM model based on availability and requirements
        self._load_vlm_model(vlm_model_name)
        
        # Deepfake detection prompts (physics & structural)
        self.analysis_prompts = {
            "physics_check": self._get_physics_prompt(),
            "structural_check": self._get_structural_prompt(),
            "uncanny_valley": self._get_uncanny_prompt(),
            "overall_assessment": self._get_overall_prompt()
        }
        
        print(f"✓ VLM Reasoner initialized successfully")
        print(f"{'='*70}\n")
    
    def _load_vlm_model(self, model_name: str):
        """Load the Vision-Language Model"""
        try:
            if self.use_lightweight:
                # Use BLIP-2 for faster inference
                print("Loading BLIP-2 (lightweight, fast inference)...")
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                
                self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                # Use InstructBLIP for better reasoning
                print(f"Loading InstructBLIP (better reasoning)...")
                from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
                
                self.processor = InstructBlipProcessor.from_pretrained(model_name)
                self.model = InstructBlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠ Error loading VLM: {e}")
            print("Falling back to rule-based heuristics...")
            self.model = None
            self.processor = None
    
    def _get_physics_prompt(self) -> str:
        """Prompt for physics consistency check"""
        return """Analyze this image carefully for physical inconsistencies:
1. Are the shadows consistent with a single light source?
2. Do reflections in mirrors, windows, or water match the scene?
3. Is the lighting direction consistent across all objects and people?
4. Are there any impossible shadow directions or missing shadows?
5. Do metallic or glass surfaces reflect the environment correctly?

Describe any physics violations you observe. Be specific about what's wrong."""
    
    def _get_structural_prompt(self) -> str:
        """Prompt for structural integrity check"""
        return """Examine this image for structural and geometric impossibilities:
1. Are there objects with impossible geometry (e.g., walls merging, perspective errors)?
2. Do furniture, buildings, or rooms have correct proportions?
3. Are there any floating objects or items defying gravity?
4. Do parallel lines converge correctly according to perspective?
5. Are there any body parts or architectural elements with impossible angles?

List any structural anomalies you find."""
    
    def _get_uncanny_prompt(self) -> str:
        """Prompt for uncanny valley detection"""
        return """Look for "uncanny valley" artifacts that make this image feel unnatural:
1. If there are faces: Are facial features proportional and natural?
2. Are skin textures too smooth, waxy, or artificial?
3. Do eyes look lifeless or have unnatural reflections?
4. Are there any unnatural symmetries or repetitive patterns?
5. Does anything look "too perfect" to be real?

Describe what feels unnatural or synthetic about this image."""
    
    def _get_overall_prompt(self) -> str:
        """Prompt for overall authenticity assessment"""
        return """Based on a careful analysis of physics, structure, and naturalness:
Is this image authentic (real photograph) or manipulated/AI-generated?
Provide your assessment and list the top 2-3 red flags that indicate manipulation.
Be concise and specific."""
    
    def analyze_image(self, image: Image.Image) -> Dict[str, any]:
        if self.model is None:
            # Fallback to heuristic analysis if VLM not available
            return self._heuristic_analysis(image)
        
        try:
            # Run all analysis prompts
            analyses = {}
            
            for check_name, prompt in self.analysis_prompts.items():
                response = self._query_vlm(image, prompt)
                analyses[check_name] = response
            
            # Synthesize results
            result = self._synthesize_analysis(analyses)
            
            return result
            
        except Exception as e:
            print(f"⚠ VLM analysis error: {e}")
            return self._heuristic_analysis(image)
    
    def _query_vlm(self, image: Image.Image, prompt: str) -> str:
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_beams=5,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return response.strip()
    
    def _synthesize_analysis(self, analyses: Dict[str, str]) -> Dict[str, any]:
        
        # Combine all analyses
        full_reasoning = []
        red_flags = []
        
        # Extract key findings from each analysis
        for check_name, response in analyses.items():
            if check_name == "overall_assessment":
                full_reasoning.insert(0, response)  # Put overall first
            else:
                if self._has_concerns(response):
                    red_flags.append(f"{check_name}: {response[:100]}...")
        
        # Determine manipulation type from red flags
        manipulation_type = self._classify_manipulation_type(analyses)
        
        # Calculate semantic score based on findings
        semantic_score = self._calculate_semantic_score(analyses, red_flags)
        
        # Create concise 2-sentence reasoning
        vlm_reasoning = self._create_reasoning_summary(analyses, red_flags)
        
        return {
            "vlm_reasoning": vlm_reasoning,
            "manipulation_type": manipulation_type,
            "semantic_score": semantic_score,
            "confidence": min(0.95, 0.6 + len(red_flags) * 0.1),  # Higher confidence with more red flags
            "detailed_analysis": analyses  # Keep for debugging
        }
    
    def _has_concerns(self, response: str) -> bool:
        """Check if response indicates concerns/red flags"""
        concern_keywords = [
            "inconsistent", "impossible", "unnatural", "synthetic", "artificial",
            "violation", "anomaly", "incorrect", "suspicious", "manipulated",
            "fake", "generated", "wrong", "defying", "too perfect"
        ]
        
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in concern_keywords)
    
    def _classify_manipulation_type(self, analyses: Dict[str, str]) -> str:
        """Classify the type of manipulation based on VLM analysis"""
        
        full_text = " ".join(analyses.values()).lower()
        
        # Classification rules based on detected issues
        if "face" in full_text or "facial" in full_text or "skin" in full_text:
            if "smooth" in full_text or "waxy" in full_text:
                return "deepfake"
            return "face_manipulation"
        
        if "shadow" in full_text or "lighting" in full_text or "reflection" in full_text:
            return "inpainting"
        
        if "geometry" in full_text or "perspective" in full_text or "structural" in full_text:
            return "composition"
        
        if "synthetic" in full_text or "generated" in full_text or "ai" in full_text:
            return "full_synthesis"
        
        if "too perfect" in full_text or "repetitive" in full_text:
            return "ai_enhancement"
        
        return "unknown_manipulation"
    
    def _calculate_semantic_score(self, analyses: Dict[str, str], red_flags: list) -> float:
        base_score = min(0.9, len(red_flags) * 0.2)
        
        # Check overall assessment for strong language
        overall = analyses.get("overall_assessment", "").lower()
        
        if "clearly manipulated" in overall or "definitely fake" in overall:
            base_score = max(base_score, 0.85)
        elif "likely manipulated" in overall or "probably fake" in overall:
            base_score = max(base_score, 0.7)
        elif "possibly manipulated" in overall or "may be fake" in overall:
            base_score = max(base_score, 0.55)
        elif "authentic" in overall or "real" in overall:
            base_score = min(base_score, 0.4)
        
        return round(base_score, 4)
    
    def _create_reasoning_summary(self, analyses: Dict[str, str], red_flags: list) -> str:
        if len(red_flags) == 0:
            return "The image appears authentic with no significant physical or structural inconsistencies detected. All elements follow expected real-world physics and geometry."
        
        # Extract top concerns
        physics_issues = []
        structural_issues = []
        uncanny_issues = []
        
        for flag in red_flags:
            if "physics" in flag:
                physics_issues.append(flag.split(": ", 1)[1] if ": " in flag else flag)
            elif "structural" in flag:
                structural_issues.append(flag.split(": ", 1)[1] if ": " in flag else flag)
            elif "uncanny" in flag:
                uncanny_issues.append(flag.split(": ", 1)[1] if ": " in flag else flag)
        
        # Build 2-sentence summary
        sentence1 = ""
        sentence2 = ""
        
        if physics_issues:
            sentence1 = f"Physics inconsistencies detected: {physics_issues[0][:80]}."
        elif structural_issues:
            sentence1 = f"Structural anomalies observed: {structural_issues[0][:80]}."
        elif uncanny_issues:
            sentence1 = f"Unnatural artifacts present: {uncanny_issues[0][:80]}."
        
        # Second sentence with additional red flag
        remaining_issues = [i for group in [structural_issues, uncanny_issues] for i in group if i not in [sentence1]]
        if remaining_issues:
            sentence2 = f"Additionally, {remaining_issues[0][:80]}."
        else:
            sentence2 = "These indicators suggest the image has been manipulated or AI-generated."
        
        return f"{sentence1} {sentence2}".strip()
    
    def _heuristic_analysis(self, image: Image.Image) -> Dict[str, any]:
        img_array = np.array(image)
        
        # Basic heuristics
        h, w = img_array.shape[:2]
        
        # Check for unnatural smoothness (common in AI faces)
        smoothness_score = self._check_smoothness(img_array)
        
        # Check for color consistency
        color_consistency = self._check_color_consistency(img_array)
        
        # Combine heuristics
        semantic_score = (smoothness_score + color_consistency) / 2
        
        # Generate reasoning
        if semantic_score > 0.6:
            reasoning = "Heuristic analysis detected unnatural smoothness and color patterns. These characteristics are common in AI-generated or heavily edited images."
            manipulation_type = "ai_enhancement"
        else:
            reasoning = "Image appears to have natural texture and color variation. No obvious manipulation artifacts detected through heuristic analysis."
            manipulation_type = "authentic"
        
        return {
            "vlm_reasoning": reasoning,
            "manipulation_type": manipulation_type,
            "semantic_score": round(semantic_score, 4),
            "confidence": 0.5,  # Lower confidence for heuristics
            "method": "heuristic_fallback"
        }
    
    def _check_smoothness(self, img_array: np.ndarray) -> float:
        from scipy.ndimage import gaussian_filter
        
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        
        # Smooth the image
        smoothed = gaussian_filter(gray, sigma=2)
        
        # Calculate difference (high = more detail, low = smooth)
        detail = np.abs(gray - smoothed)
        avg_detail = np.mean(detail)
        
        # Low detail suggests AI smoothing
        if avg_detail < 5:
            return 0.8  # Likely AI-generated
        elif avg_detail < 10:
            return 0.5
        else:
            return 0.2  # Likely real
    
    def _check_color_consistency(self, img_array: np.ndarray) -> float:
        """Check for unnatural color patterns"""
        
        # Calculate color histogram entropy
        from scipy.stats import entropy
        
        hist_r = np.histogram(img_array[:,:,0], bins=32)[0]
        hist_g = np.histogram(img_array[:,:,1], bins=32)[0]
        hist_b = np.histogram(img_array[:,:,2], bins=32)[0]
        
        # Normalize
        hist_r = hist_r / hist_r.sum()
        hist_g = hist_g / hist_g.sum()
        hist_b = hist_b / hist_b.sum()
        
        # Calculate entropy (low entropy = less natural)
        avg_entropy = (entropy(hist_r) + entropy(hist_g) + entropy(hist_b)) / 3
        
        # Typical images have entropy > 3.5
        if avg_entropy < 3.0:
            return 0.7  # Suspicious
        elif avg_entropy < 3.5:
            return 0.4
        else:
            return 0.2  # Looks natural

def test_vlm_reasoner():
    
    print("\n" + "="*70)
    print("TESTING MODULE 2: VLM LOGIC REASONER")
    print("="*70)
    
    # Initialize reasoner
    reasoner = VLMLogicReasoner(use_lightweight=True)
    
    # Create a dummy test image
    test_image = Image.new('RGB', (512, 512), color='white')
    
    print("\nRunning analysis on test image...")
    result = reasoner.analyze_image(test_image)
    print("ANALYSIS RESULTS:")
    print(f"VLM Reasoning: {result['vlm_reasoning']}")
    print(f"Manipulation Type: {result['manipulation_type']}")
    print(f"Semantic Score: {result['semantic_score']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    return reasoner


if __name__ == "__main__":
    reasoner = test_vlm_reasoner()