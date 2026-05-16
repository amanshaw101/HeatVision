# Weak Supervision for Urban Heat Island Segmentation Using Physics-Derived Pseudo-Labels

A physics-guided deep learning approach that automatically detects Urban Heat Islands (UHI) from raw Landsat satellite imagery — without requiring any manual annotation.

## Abstract

We present a framework for UHI detection that combines physics-based remote sensing with weakly supervised deep learning. Starting from raw Landsat 8/9 Level-1 data, we derive Land Surface Temperature (LST) from first principles and generate training labels automatically using physics-guided heuristics (temperature, vegetation index, and surface reflectance thresholds). A U-Net trained with a **physics-informed loss function** embeds domain constraints directly into the learning process. We evaluate across 5 cities spanning different climates and validate using multiple independent strategies.

## Key Contributions

1. **Physics-guided weak labeling**: Auto-generate UHI training labels from LST, NDVI, and surface reflectance — no manual annotation needed
2. **Physics-informed loss function**: Embed vegetation and temperature constraints directly in the neural network's training objective
3. **Desert-aware labeling**: Use Band 4 reflectance to distinguish urban surfaces from desert sand
4. **Multi-strategy validation**: Break circular evaluation with alternative labels, LST correlation, and leave-one-city-out testing

---

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

### What to Download
- **Landsat 8-9 OLI/TIRS C2 L1** data from [USGS EarthExplorer](https://earthexplorer.usgs.gov/)
- **Level-1** (raw DN) — we derive LST from first principles
- **Summer scenes** with **<10% cloud cover**

### Files Needed (per city)
Only 4 files from each Landsat scene:
1. `*_B4.TIF` (Red)
2. `*_B5.TIF` (NIR)
3. `*_B10.TIF` (Thermal)
4. `*_MTL.txt` (Metadata)

### Directory Structure
```
data/
  London/
    LC08_.../
      *_B4.TIF, *_B5.TIF, *_B10.TIF, *_MTL.txt
  Dubai/
    ...
  Mumbai/
    ...
  Beijing/    (optional: add more cities)
    ...
  Cairo/
    ...
```

## Pipeline

### 1. Preprocessing
Generate LST, NDVI, and weak labels for each city:
```bash
python src/preprocessing.py --city London
python src/preprocessing.py --city Dubai
python src/preprocessing.py --city Mumbai
```

### 2. Training
Train with **physics-informed loss** (recommended):
```bash
python src/train.py --epochs 20 --loss physics
```

Leave-one-city-out training:
```bash
python src/train.py --epochs 20 --loss physics --exclude-city London
```

### 3. Evaluation
```bash
python src/evaluate.py
```
Outputs `evaluation_metrics.json` with AUPRC, ROC-AUC, IoU, F1, R², RMSE per city.

### 4. Validation
Break circular evaluation:
```bash
python src/validate.py --strategy all         # Run all validation strategies
python src/validate.py --strategy lst-correlation
python src/validate.py --strategy alt-labels
python src/validate.py --strategy leave-one-out --epochs 10
```
Outputs `validation_results.json`.

### 5. Interactive Visualization
```bash
streamlit run app.py
```

## Project Structure

```text
├── UHI_IEEE_Manuscript.docx # Formatted IEEE research manuscript with reviewer feedback
├── app.py                  # Streamlit UI (3 tabs: Analysis, Upload, Metrics)
├── src/
│   ├── preprocessing.py    # L1 DN → LST → Weak Labels
│   ├── model.py            # U-Net architecture
│   ├── dataset.py          # Landsat dataset with physics data
│   ├── loss.py             # BCE, Dice, CombinedLoss, PhysicsInformedLoss
│   ├── train.py            # Training with loss/city/scheduler options
│   ├── evaluate.py         # Metrics: AUPRC, ROC-AUC, IoU, F1, R², RMSE
│   └── validate.py         # Validation framework (3 strategies)
├── data/                   # City satellite data
├── checkpoints/            # Model weights
└── requirements.txt
```

## Methodology

1. **Raw DN → LST**: Radiometric calibration → Brightness Temperature → NDVI → Emissivity → Land Surface Temperature
2. **Weak Label Generation**: `UHI = (LST > μ + 0.5σ) ∧ (0 < NDVI < 0.3) ∧ (Red Refl < 0.25) ∧ (LST > 200K)`
3. **Physics-Informed Training**: `L = 0.5·BCE + 0.5·Dice + λ_veg·L_vegetation + λ_temp·L_temperature`
4. **Post-Processing**: Morphological opening + physics constraint masking
5. **Multi-Strategy Validation**: Alternative labels, LST correlation, leave-one-city-out

## Recent Updates & Progress

- **IEEE Manuscript Preparation**: The project research has been formatted into an IEEE conference manuscript (`UHI_IEEE_Manuscript.docx`), incorporating comprehensive responses to peer review feedback (e.g., deeper justifications for physics-derived labels, baseline comparison tables, and detailed implementation specifications).
- **Streamlit UI Refinement**: The interactive dashboard (`app.py`) was cleaned up to improve UX. The "Validation Results" tab and cluttered metrics were removed, streamlining the interface to 3 core tabs: Interactive Analysis, Upload Data, and Research Metrics. We also laid the groundwork for a future tactical, data-dense "Satellite Command" UI.
- **Enhanced Documentation**: Added clear, humanized documentation outlining the end-to-end technical mapping pipeline, data collection process, and how heuristic formulas interact with the U-Net architecture.
- **Showcase & Presentation**: Extracted key code snippets demonstrating the physics-informed loss functions and unique data processing logic for project presentations.
