# M&Ms Cardiac MRI Segmentation Baseline

## Overview

This repository contains a baseline implementation for multi-class cardiac MRI segmentation on the M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation) dataset.

The baseline model is a 2D MONAI U-Net trained for segmentation of:

- Left Ventricle (LV)
- Myocardium
- Right Ventricle (RV)

The project serves as the baseline for future work on prompt-driven domain generalization.

---

## Dataset

Dataset: M&Ms Cardiac MRI Dataset

Validation split contains ground-truth annotations and is used for quantitative evaluation.

---

## Model

- MONAI 2D U-Net
- Input size: 256 × 256
- Output classes: 4
  - Background
  - LV
  - Myocardium
  - RV

---

## Evaluation

Evaluation is performed at the patient/volume level.

Metrics:

- Dice (LV, Myocardium, RV)
- Mean Dice
- HD95 (LV, Myocardium, RV)
- Mean HD95

---

## Baseline Results

Overall validation performance:

| Metric | Value |
|---------|------:|
| Mean Dice | 0.8100 |
| Mean HD95 | 18.81 mm |

---

## Additional Analysis

Implemented:

- Patient-level evaluation
- Per-class Dice
- Per-class HD95
- ED vs ES evaluation
- Vendor-wise evaluation
- Qualitative visualization
- Failure case analysis

---

## Repository Structure

```
src/
    dataset.py
    train.py
    evaluate_volume.py
    qualitative_results.py
    phase_analysis.py
    vendor_analysis.py

volume_evaluation_results/
    patient_level_results.csv
    patient_summary.csv
    phase_summary.csv
    vendor_summary.csv

qualitative_results/
    D1R0Y5_ED.png
    C8I7P7_ED.png
    P8V0Y7_ES.png
```

---

## Running

Train

```bash
python train.py
```

Patient-level evaluation

```bash
python evaluate_volume.py
```

Phase analysis

```bash
python phase_analysis.py
```

Vendor analysis

```bash
python vendor_analysis.py
```

Qualitative visualization

```bash
python qualitative_results.py
```

---

## Future Work

- Leave-one-vendor experiments
- Leave-one-centre experiments
- Prompt-driven domain generalization
- Latent domain clustering