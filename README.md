# 🫁 Chest X-Ray Pneumonia Detection using Deep Learning

<div align="center">

<img width="737" height="305" alt="image" src="https://github.com/user-attachments/assets/cf9332b8-2261-46fe-9b31-865bb56fd08e" />

<img width="554" height="709" alt="image" src="https://github.com/user-attachments/assets/cb9b6688-f22b-4cc3-a36b-a43be35f17aa" />

<img width="551" height="740" alt="image" src="https://github.com/user-attachments/assets/5dea4d23-6e77-404d-97e6-7e662cada83c" />

<img width="548" height="714" alt="image" src="https://github.com/user-attachments/assets/7a920734-b842-484a-9286-9e86aaf16641" />

<img width="686" height="355" alt="image" src="https://github.com/user-attachments/assets/22542c1e-0fb7-404a-a4ff-f1bd085dab07" />


**An explainable, clinically-aware AI system for pneumonia detection from chest X-rays, combining ensemble deep learning, Grad-CAM visualisation, and neuro-symbolic reasoning.**

[Overview](#-overview) • [Architecture](#-architecture) • [Results](#-results) • [Setup](#-setup) • [Usage](#-usage) • [Future Work](#-future-enhancements)

</div>

---

## 📋 Overview

Pneumonia is a life-threatening respiratory infection that disproportionately affects children under five and the elderly. Diagnosis via chest X-rays requires significant radiological expertise — a resource that is scarce in many healthcare settings.

This project proposes a **multi-component AI pipeline** that goes beyond simple binary classification by incorporating:

- ✅ **Ensemble Learning** — EfficientNetV2B0 + DenseNet121 for complementary feature extraction
- ✅ **Grad-CAM Explainability** — Heatmaps showing *why* the model made a decision
- ✅ **Neuro-Symbolic Reasoning** — Lukasiewicz t-norm fuzzy logic encoding 6 radiological rules
- ✅ **Clinical Decision Support** — Severity scores, confidence levels, and uncertainty flagging
- ✅ **Streamlit Deployment** — Ready-to-use web interface for single-image clinical reporting

> 📄 *Bachelor of Technology (CSE - Data Science) Final Year Project — VIT Vellore, April 2026*
> *Supervised by Prof. Saira Banu J*

---

## 🏗️ Architecture

The system is organised as a **4-stage hierarchical pipeline**:

```
Chest X-Ray (JPEG/PNG)
        │
        ▼
┌─────────────────────────────────┐
│   Stage 1: Data Pipeline        │
│   tf.data • Augmentation        │
│   Class Balancing • Focal Loss  │
└──────────────┬──────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│EfficientNet │  │ DenseNet121 │
│   V2B0      │  │             │
│  p_eff ──── │  │ ──── p_den  │
└─────────────┘  └─────────────┘
       │               │
       └───────┬───────┘
               ▼
┌─────────────────────────────────┐
│   Stage 3: Weighted Ensemble    │
│   w=0.4×AUC + 0.6×Recall       │
│   Threshold Optimisation (0.79) │
└──────────┬──────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────┐
│ Grad-CAM│  │  Neuro-Symbolic  │
│Heatmaps │  │  Reasoning Layer │
│         │  │  6 Fuzzy Rules   │
└─────────┘  └──────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Clinical Report (Streamlit)    │
│  Prediction • Severity • Action │
└─────────────────────────────────┘
```

### Model Details

| Component | Details |
|-----------|---------|
| **Primary Backbone** | EfficientNetV2B0 (ImageNet pretrained, two-phase fine-tuning) |
| **Secondary Backbone** | DenseNet121 (ImageNet pretrained, full fine-tuning) |
| **Loss Function** | Focal Loss (γ=2.0, α=0.25) |
| **Optimizer** | Adam (LR=1e-4 → 1e-5) |
| **Input Size** | 224×224×3 |
| **Ensemble Strategy** | AUC-weighted (40%) + Recall-weighted (60%) |
| **Optimal Threshold** | 0.79 (F1-maximising) |
| **Explainability** | Grad-CAM (last convolutional layer) |
| **Symbolic Layer** | Lukasiewicz t-norm fuzzy logic (6 rules) |
| **Training Platform** | Kaggle — Dual NVIDIA Tesla T4 GPU |

---

## 📊 Results

### Individual Model Performance (Threshold = 0.50)

| Metric | EfficientNetV2B0 | DenseNet121 |
|--------|:---:|:---:|
| Accuracy | 85% | 91% |
| AUC-ROC | 0.960 | 0.968 |
| Pneumonia Recall | 0.98 | 0.99 |
| Normal Recall | 0.65 | 0.79 |
| Macro F1 | 0.83 | 0.90 |

### Ensemble Performance Comparison

| Configuration | Accuracy | Normal Recall | Pneumonia Recall | Macro F1 |
|---------------|:---:|:---:|:---:|:---:|
| Average (τ=0.50) | 90% | 0.74 | 0.99 | 0.88 |
| Weighted (τ=0.50) | 90% | 0.74 | 0.99 | 0.88 |
| **Weighted (τ=0.79)** | **93%** | **0.91** | **0.94** | **0.93** |

### Neuro-Symbolic Reasoner (Full Test Set)

| Metric | NORMAL | PNEUMONIA | Overall |
|--------|:---:|:---:|:---:|
| Precision | 0.90 | 0.94 | 0.92 |
| Recall | 0.90 | 0.94 | 0.92 |
| F1-Score | 0.90 | 0.94 | 0.92 |
| **Accuracy** | | | **92%** |

> 🔍 **68.4%** of predictions made with HIGH confidence • **13.3%** flagged for mandatory radiologist review

---

## 🧠 Neuro-Symbolic Rules

The reasoning layer encodes **6 clinical radiological rules** using Lukasiewicz t-norm fuzzy logic:

| Rule | Clinical Basis | Logic |
|------|----------------|-------|
| R1 | Consolidation OR patchy infiltrates → pneumonia | `L_or(c1, c3)` |
| R2 | Ground-glass opacity → pneumonia | `c2` |
| R3 | Air bronchograms AND (consolidation OR GGO) → pneumonia | `L_and(c4, L_or(c1, c2))` |
| R4 | Pleural effusion → pneumonia | `c5` |
| R5 | Hyperinflation → pneumonia (viral) | `c6` |
| C  | Coherence: all concepts low → prediction low | `p_ens × (1 − max_concept)` |

```python
# Lukasiewicz T-norm operations
L_and(a, b) = max(0.0, a + b - 1.0)   # Fuzzy conjunction
L_or(a, b)  = min(1.0, a + b)          # Fuzzy disjunction
```

---

## 📁 Project Structure

```
📦 pneumonia-detection/
├── 📂 data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   └── test/
├── 📂 models/
│   ├── efficientnet_model.h5
│   └── densenet_model.h5
├── 📂 src/
│   ├── data_pipeline.py        # tf.data pipeline with augmentation
│   ├── focal_loss.py           # Custom focal loss implementation
│   ├── train_efficientnet.py   # EfficientNetV2B0 training (2-phase)
│   ├── train_densenet.py       # DenseNet121 training
│   ├── ensemble.py             # Weighted ensemble + threshold tuning
│   ├── gradcam.py              # Grad-CAM heatmap generation
│   ├── neuro_symbolic.py       # Lukasiewicz t-norm rule engine
│   └── reasoner.py             # Clinical decision ladder
├── 📂 notebooks/
│   └── full_pipeline.ipynb
├── app.py                      # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
opencv-python>=4.8.0
streamlit>=1.28.0
Pillow>=10.0.0
```

### Dataset

Download the **Kaggle Chest X-Ray Pneumonia Dataset**:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

Dataset contains **5,216 images** across train/val/test splits:
- **Train**: 1,341 NORMAL • 3,875 PNEUMONIA
- **Val**: 8 NORMAL • 8 PNEUMONIA
- **Test**: 234 NORMAL • 390 PNEUMONIA

---

## 🚀 Usage

### Training

```bash
# Train EfficientNetV2B0 (2-phase transfer learning)
python src/train_efficientnet.py --data_dir ./data --epochs_phase1 15 --epochs_phase2 8

# Train DenseNet121 (full fine-tuning)
python src/train_densenet.py --data_dir ./data --epochs 15

# Build ensemble and tune threshold
python src/ensemble.py --eff_model ./models/efficientnet_model.h5 \
                       --den_model ./models/densenet_model.h5 \
                       --test_dir ./data/test
```

### Inference

```python
from src.reasoner import NeuroSymbolicReasoner

# Load reasoner
reasoner = NeuroSymbolicReasoner(
    eff_model_path='models/efficientnet_model.h5',
    den_model_path='models/densenet_model.h5'
)

# Predict on a single image
result = reasoner.predict(img_array)

print(f"Prediction:    {result['prediction']}")
print(f"Confidence:    {result['confidence_level']}")
print(f"Severity:      {result['severity_score']}/10")
print(f"Top Feature:   {result['dominant_feature']}")
print(f"Action:        {result['recommended_action']}")
print(f"Models Agree:  {result['models_agree']}")
```

### Streamlit App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` and upload a chest X-ray to get an instant clinical report with Grad-CAM heatmap.

---

## 🔬 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Focal Loss over BCE** | Handles 2.89:1 class imbalance by down-weighting easy examples |
| **Two-phase training for EfficientNet** | Avoids catastrophic forgetting; head trained first, top-20 layers fine-tuned at 10× lower LR |
| **Full fine-tuning for DenseNet** | Dense connectivity requires all layers to cooperate — frozen early layers create stale concatenated features |
| **Threshold 0.79 (not 0.50)** | Improves NORMAL recall from 0.74 → 0.91 at modest cost to PNEUMONIA recall (0.99 → 0.94) |
| **Recall-weighted ensemble (60%)** | Clinical priority: missing pneumonia (FN) is more dangerous than a false alarm (FP) |
| **Lukasiewicz t-norm** | Differentiable fuzzy logic — rules can be incorporated as loss terms in future end-to-end training |

---

## 🔭 Future Enhancements

- [ ] **Lung Segmentation** — U-Net preprocessing to isolate lung parenchyma
- [ ] **3-class Classification** — Normal / Bacterial Pneumonia / Viral Pneumonia
- [ ] **Uncertainty Quantification** — Monte Carlo Dropout for calibrated confidence intervals
- [ ] **Multi-Dataset Validation** — CheXpert, MIMIC-CXR, Indiana University dataset
- [ ] **Concept Bottleneck Models** — Explicit concept-level predictors replacing proxy scoring
- [ ] **Federated Learning** — Privacy-preserving training across multiple clinical institutions
- [ ] **Differentiable Rule Integration** — Rule loss as training penalty: `L_total = L_focal + λ × L_rules`

---

## 📚 References

- Rajpurkar et al. (2017) — *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays*
- Huang et al. (2017) — *Densely Connected Convolutional Networks*
- Tan & Le (2021) — *EfficientNetV2: Smaller Models and Faster Training*
- Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- Kermany et al. (2018) — *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning* (Cell)

---

## 👨‍💻 Author

**Hitansh Sondhi** (22BDS0325)
B.Tech — Computer Science and Engineering (Data Science)
Vellore Institute of Technology, Vellore

*Supervised by Prof. Saira Banu J, Professor Grade 1, SCOPE — VIT*

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

⭐ If you found this project useful, please consider giving it a star!

</div>
