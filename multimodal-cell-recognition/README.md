# Label-free cell recognition with optical multimodal imaging

Whole-field registration, Cellpose segmentation, multimodal CNN classification, and analysis of 15 optical proxies.

## Setup

```bash
conda create -n celloptics python=3.10
conda activate celloptics
pip install -r requirements.txt
```

Use a PyTorch 2.5.1 build compatible with your CUDA driver. Pretrained weights are downloaded on first use.

## Data and configuration

Place `whole_field_data/` beside this repository. It contains 396 field triplets (1,188 images), named `CellClass-Modality-ID.jpg`, for example `MCF-7-BFR-0001.jpg`.

- `configs/fields.json`: image hashes, A/B/C partitions, and parent-field CV folds.
- `configs/protocol.json`: model settings, fixed seeds, and statistics.
- `configs/proxies.json`: p01–p15 definitions and feature dimensions.


## Run

Run commands from the repository root. Results are saved to `outputs/`; use `python run.py --output /path/to/results <task>` for a custom location. Use a new output directory after changing code or configuration.

### Prepare whole-field images

```bash
python run.py verify-data
python run.py prepare --data ../whole_field_data
```

Preparation registers BFR/BFT to DFR, segments cells with Cellpose cyto3, applies quality filtering, and exports masks and crops. Classification uses target-masked crops.

### CNN classification

```bash
python run.py screen
python run.py select
python run.py fusion
python run.py evaluate --all-backbones
python run.py statistics cnn
python run.py statistics confidence
```

Six backbones are compared using mean validation macro-F1 across three parent-field-grouped folds within A+B. Selected fold models are probability-ensembled on C. Training uses 224 × 224 inputs, AdamW, batch size 16, and up to 30 epochs.

### Proxy classification and interventions

```bash
python run.py proxy-features
python run.py proxy-classify
python run.py proxy-evaluate
python run.py statistics proxy
python run.py proxy-models
python run.py intervene
python run.py statistics interventions
```

The 15 shared proxies produce 180 features per modality.

Proxy classifiers and intervention CNNs use A for training, B for selection, and C for evaluation. Interventions use four doses (25%, 50%, 75%, 100%) and eight RMS-matched controls per dose. Specificity is control minus targeted true-class probability; R is its dose–response integral. BH correction covers 45 R tests.

### Representation analysis and checks

```bash
python run.py representations
python run.py gradcam --sample-id <test-cell-id>
python -m unittest discover -s tests
```

These commands export PCA/t-SNE coordinates and single-modality Grad-CAM arrays. Numerical outputs include predictions, metrics, confusion matrices, and bootstrap statistics. Seeds are fixed, and selected models are frozen before test evaluation.
