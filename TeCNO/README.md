# TeCNO Cholec80 Compression Robustness

This folder contains the TeCNO-based surgical phase recognition pipeline used for evaluating the effect of video compression on the Cholec80 dataset.

## Project Overview

The goal of this experiment is to test how TeCNO performs when trained and tested across different video compression levels.

The compression levels used are:

- Original
- CRF18
- CRF23
- CRF28
- CRF35
- CRF51

## Pipeline

The pipeline has five main steps:

1. Preprocess Cholec80 videos at 1 fps
2. Train Stage 1 ResNet50 feature extractor
3. Export features across compression levels
4. Train Stage 2 TeCNO/MS-TCN temporal model
5. Evaluate cross-compression performance and generate plots

## Important Files

### Preprocessing

- `preprocessing_1fps.py`

This script converts Cholec80 videos into 1 fps extracted frames and creates the dataframe `.pkl` file used for training.

### Stage 1

- `stage1_train_phase_pretrained.py`
- `stage1_export_features_all_pretrained.py`
- `stage1_export_features_cross.py`

These scripts train the pretrained ResNet50-based feature extractor and export frame-level features.

### Stage 2

- `stage2_train_full.py`
- `stage2_test_only.py`

These scripts train and test the TeCNO temporal model using the exported features.

### Evaluation

- `make_tecno_plots.py`

This script generates the result plots and heatmaps for cross-compression evaluation.

## Main Modifications

Compared to the original TeCNO codebase, this version adds:

- 1 fps preprocessing for Cholec80 videos
- Pretrained ResNet50 feature extraction
- Cross-compression feature export
- Stage 2 cross-compression testing
- Automated batch testing script
- Result plotting for accuracy and F1 score

## Notes

Generated folders such as `logs/`, `wandb/`, checkpoints, exported features, and raw video frames are not included in this repository.