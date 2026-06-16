# Weakly-Supervised Remote Sensing Segmentation
### Partial Cross-Entropy Loss with SegFormer on LoveDA

A deep learning assignment implementing **weakly-supervised semantic segmentation** for remote sensing imagery using sparse point annotations instead of full dense labels.

---

## 📌 Project Overview

This project trains a **SegFormer-B2** model on the **LoveDA** urban/rural land cover dataset using only **point-level annotations** — a weakly-supervised approach that dramatically reduces labeling cost.

The core contribution is a **Partial Cross-Entropy (PCE) Loss** (with a Focal variant) that supervises only the sparsely labeled pixels:

$$\text{pfCE} = \frac{\sum \text{FocalLoss}(\hat{y}, y) \times \text{MASK}_{\text{labeled}}}{\sum \text{MASK}_{\text{labeled}}}$$

### Assessment Tasks
1. Implement Partial Cross-Entropy Loss (standard CE + Focal variant)
2. Simulate sparse point labels from full LoveDA masks → train SegFormer
3. Ablation experiments:
   - **(A)** Effect of point annotation density
   - **(B)** CE loss vs. Focal loss comparison

---

## 🗺️ Dataset — LoveDA

**LoveDA** (Love Data) is a remote sensing land cover segmentation dataset with Urban and Rural scenes.

| Class ID | Class Name  | Color |
|----------|-------------|-------|
| 0 | Background  | White |
| 1 | Building    | Red |
| 2 | Road        | Gray |
| 3 | Water       | Blue |
| 4 | Barren      | Tan |
| 5 | Forest      | Green |
| 6 | Agriculture | Yellow |

---

## 🏗️ Model Architecture

- **Backbone:** SegFormer-B2 (Mix Transformer) — pretrained on ImageNet via HuggingFace `transformers`
- **Head:** Lightweight MLP decode head
- **Output:** 7-class semantic segmentation map (upsampled from 1/4 resolution)

---

## 🔑 Key Components

### 1. Focal Loss
```python
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```
Downweights easy pixels and focuses training on hard, misclassified examples.

### 2. Partial Cross-Entropy Loss (`PartialCELoss`)
Computes element-wise CE/Focal loss, **zeros out unlabeled pixels** using the point mask, and normalizes only over labeled pixels.

### 3. Point Annotation Simulator (`PointAnnotationSimulator`)
Converts a full dense mask to sparse point annotations by sampling `K` random pixels per class present in the image. Worker-safe for use with PyTorch `DataLoader`.

### 4. LoveDA Dataset (`LoveDADataset`)
Handles multiple Kaggle directory layouts automatically. Remaps mask values (1–7 → 0–6, void → 255).

### 5. Metrics
Streaming **mIoU** and per-class IoU via confusion matrix accumulation.

---

## 📁 Repository Structure

```
Assignment/
├── code.ipynb      # Full implementation: model, loss, training, ablations, visualizations
├── Report.pdf      # Written report with methodology, results, and analysis
└── README.md       # Project documentation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| PyTorch | Deep learning framework |
| HuggingFace `transformers` | SegFormer-B2 pretrained model |
| `albumentations` | Image augmentation |
| `matplotlib` / `seaborn` | Visualization |
| `wandb` | Experiment tracking |
| Jupyter Notebook | Interactive development |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/nimradev064/Assignment.git
cd Assignment
```

### 2. Install dependencies
```bash
pip install transformers accelerate timm albumentations matplotlib seaborn wandb
pip install segmentation-models-pytorch
```

### 3. Prepare the LoveDA dataset
Download from Kaggle and place it in the expected structure:
```
loveda-dataset/
  Train/Urban/images_png/
  Train/Urban/masks_png/
  Train/Rural/images_png/
  Train/Rural/masks_png/
  Val/...
```

### 4. Run the notebook
```bash
jupyter notebook code.ipynb
```

---

## 📊 Experiments & Results

The notebook includes:
- **Training curves** (loss & accuracy per epoch)
- **Per-class IoU heatmap** across all 7 LoveDA classes
- **Qualitative visualizations**: input image, ground-truth mask, and model prediction side-by-side
- **Ablation study**: point density (1, 3, 5, 10 pts/class) and CE vs. Focal loss

---

## 📄 Report

See `Report.pdf` for the full written report covering:
- Problem statement and motivation
- Methodology (PCE loss formulation, point simulation strategy)
- Quantitative results and ablation analysis
- Conclusions

---


## 📜 License

This project is submitted for academic assessment purposes.
