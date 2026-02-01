# Deepfake Detection System - Competition Submission 🏆

**Competition:** Detecting GenAI & Sophisticated Manipulation in Public Media  
**MenaML Winter School 2026**

## 📋 Overview

This system combines two complementary detection modules:

1. **Module 1: Forensic Signal Detector** - Pixel-level analysis using EfficientX3D-XS
2. **Module 2: VLM Logic Reasoner** - Semantic analysis using BLIP-2/InstructBLIP

**Performance:** F1 Score = 0.9314 on validation set

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Inference

```bash
python predict.py \
  --input_dir /path/to/test_images \
  --output_file predictions.json \
  --checkpoint module1_checkpoints/module1-forensic-epoch=01-val_f1=0.9314.ckpt \
  --device cuda
```

**Arguments:**
- `--input_dir`: Directory containing test images (required)
- `--output_file`: Output JSON path (default: predictions.json)
- `--checkpoint`: Path to Module 1 checkpoint (required)
- `--device`: cuda or cpu (default: cuda if available)
- `--lightweight`: Use lightweight VLM for faster inference (optional)

---

## 📁 Repository Structure

```
.
├── predict.py                    # Main inference script
├── module2_vlm_reasoner.py       # Module 2 implementation
├── requirements.txt               # Dependencies
├── TECHNICAL_REPORT.md           # 3-page technical report
├── README.md                     # This file
└── module1_checkpoints/
    └── module1-forensic-epoch=01-val_f1=0.9314.ckpt  # Trained weights
```

---

## 🎯 Output Format

The system generates a JSON file with predictions for each image:

```json
[
  {
    "image_name": "000001.jpg",
    "authenticity_score": 0.91,
    "manipulation_type": "inpainting",
    "vlm_reasoning": "The window reflection is inconsistent with the room layout. Shadow direction on the sofa does not match the light source."
  },
  {
    "image_name": "000002.jpg",
    "authenticity_score": 0.15,
    "manipulation_type": "authentic",
    "vlm_reasoning": "The image appears authentic with no significant physical or structural inconsistencies detected. All elements follow expected real-world physics and geometry."
  }
]
```

**Fields:**
- `authenticity_score`: 0.0 (authentic) to 1.0 (manipulated)
- `manipulation_type`: Category of manipulation detected
- `vlm_reasoning`: 2-sentence explanation of findings

---

## 🏗️ System Architecture

### Module 1: Forensic Signal Detector

**Model:** EfficientX3D-XS (3D CNN for temporal analysis)

**Detection Capabilities:**
- GAN/Diffusion fingerprints
- Compression artifacts
- Texture inconsistencies
- Temporal anomalies

**Training:**
- Dataset: Celeb-DF v2 (590 real + 5,639 fake videos)
- F1 Score: 0.9314
- Class-weighted loss for imbalanced data

**Image Adaptation:**
Input images are converted to pseudo-videos by replicating frames 13 times to match the video model's expected input format.

### Module 2: VLM Logic Reasoner

**Model:** BLIP-2 (2.7B) or InstructBLIP (7B)

**Detection Capabilities:**
- Physics consistency (shadows, lighting, reflections)
- Structural integrity (geometry, perspective)
- Uncanny valley artifacts
- Contextual anomalies

**Analysis Types:**
1. Physics check: Light sources, shadows, reflections
2. Structural check: Geometry, proportions, gravity
3. Uncanny valley: Facial features, skin textures, symmetry

### Fusion Strategy

**Method:** Confidence-weighted ensemble

```python
# Base weights
forensic_weight = 0.6
semantic_weight = 0.4

# Dynamic adjustment based on confidence
final_score = weighted_average(
    forensic_score * (1 + forensic_confidence * 0.3),
    semantic_score * (1 + semantic_confidence * 0.3)
)
```

---

## 📊 Performance Metrics

### Module 1 (Validation Set)
- **F1 Score:** 0.9314
- **Accuracy:** 93.8%
- **Precision:** 0.94
- **Recall:** 0.92
- **AUROC:** 0.978

### Inference Speed (NVIDIA GPU)
- Module 1: ~150ms per image
- Module 2: ~800ms per image
- **Total:** ~1 second per image

---

## 🔧 Technical Details

### Module 1 Preprocessing
```python
Input Size: 182 × 182
Normalization: mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]
Frame Replication: 1 → 13 frames (for video model)
```

### Module 2 Prompts
**Physics Check:**
> "Analyze this image carefully for physical inconsistencies: Are the shadows consistent with a single light source? Do reflections match the scene?"

**Structural Check:**
> "Examine this image for structural and geometric impossibilities: Are there objects with impossible geometry? Do proportions make sense?"

**Uncanny Valley:**
> "Look for artifacts that make this image feel unnatural: Are facial features proportional? Does skin texture look artificial?"

---

## 🎓 Competition Requirements Checklist

- ✅ **Inference Script:** `predict.py`
- ✅ **Dependencies:** `requirements.txt`
- ✅ **Output Format:** JSON with required fields
- ✅ **Module 1:** Forensic Signal Detector (pixel-level)
- ✅ **Module 2:** VLM Logic Reasoner (semantic-level)
- ✅ **Technical Report:** 3-page architecture summary
- ✅ **Explainability:** Natural language reasoning
- ✅ **Command Line Interface:** Standard format

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use CPU instead
python predict.py --input_dir /test_images --device cpu

# Or use lightweight VLM
python predict.py --input_dir /test_images --lightweight
```

**2. VLM Not Loading**
```bash
# The system will automatically fall back to heuristic analysis
# Check transformers version: pip install transformers>=4.30.0
```

**3. Checkpoint Not Found**
```bash
# Ensure checkpoint path is correct
python predict.py --checkpoint /path/to/checkpoint.ckpt --input_dir /test_images
```

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{deepfake_detection_2026,
  title={Dual-Module Deepfake Detection System},
  author={[Your Team Name]},
  year={2026},
  howpublished={MenaML Winter School Competition}
}
```

---

## 👥 Team Information

**Team Name:** [Your Team Name]  
**Members:**
1. [Member 1 Name]
2. [Member 2 Name]
3. [Member 3 Name]

**Affiliation:** MenaML Winter School 2026

---

## 📧 Contact

For questions or issues, please contact: [your-email@example.com]

---

## 📄 License

This project is submitted for the MenaML Winter School 2026 Competition.

---

**Last Updated:** January 28, 2026
