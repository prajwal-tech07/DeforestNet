# DeforestNet

Satellite-based deforestation detection using deep learning. Uses Sentinel-1 (SAR) and Sentinel-2 (optical) imagery to perform binary semantic segmentation — classifying pixels as **deforestation** or **non-deforestation** in the Brazilian Amazon.

## Project Structure

```
AI_PROJECT/
├── dataset/                          # Raw .tif files (not in repo — see Setup)
│   ├── 1_CLOUD_FREE_DATASET/
│   ├── 2_CLOUDY_DATASET/
│   └── 3_TRAINING_MASKS/
└── DeforestNet/                      # This repo
    ├── configs/config.py             # Central configuration
    ├── src/preprocessing/
    │   ├── reader.py                 # GeoTIFF reader
    │   ├── noise_removal.py          # Lee speckle filter, Gaussian smoothing
    │   ├── normalization.py          # Percentile-based normalization
    │   ├── feature_extraction.py     # NDVI, EVI, SAVI, VV/VH, RVI
    │   ├── patch_extractor.py        # Patch extraction & balancing
    │   └── pipeline.py               # Main preprocessing pipeline
    ├── outputs/preprocessed/         # Generated .npz files (not in repo)
    ├── requirements.txt
    └── README.md
```

## Setup on a New Machine

### 1. Clone the repo

```bash
git clone https://github.com/prajwal-tech07/DeforestNet.git
```

### 2. Create a parent folder and place the repo inside

```
mkdir AI_PROJECT
mv DeforestNet AI_PROJECT/
cd AI_PROJECT
```

### 3. Download the dataset

Download the satellite imagery dataset and extract it so the folder structure looks like:

```
AI_PROJECT/
├── dataset/
│   ├── 1_CLOUD_FREE_DATASET/
│   │   ├── 1_SENTINEL1/
│   │   └── 2_SENTINEL2/
│   ├── 2_CLOUDY_DATASET/
│   │   ├── 1_SENTINEL1/
│   │   └── 2_SENTINEL2/
│   └── 3_TRAINING_MASKS/
│       ├── MASK_16_GRID/
│       └── MASK_FULL/
└── DeforestNet/
```

### 4. Install dependencies

```bash
cd DeforestNet
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 5. Run preprocessing

```bash
python src/preprocessing/pipeline.py
```

This reads all `.tif` files, applies noise removal, normalization, feature extraction, and creates train/val/test splits as compressed `.npz` files in `outputs/preprocessed/`.

## Preprocessing Pipeline

| Step                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| 1. Read GeoTIFF       | Loads Sentinel-1 (VV, VH), Sentinel-2 (B2, B3, B4, B8), masks |
| 2. Noise Removal      | Lee speckle filter (SAR), Gaussian smoothing (optical)        |
| 3. Normalization      | Percentile-based scaling to [0, 1] with global statistics     |
| 4. Feature Extraction | NDVI, EVI, SAVI, VV/VH ratio, RVI → 11 feature bands          |
| 5. Patch Extraction   | 256×256 non-overlapping patches, class balancing              |
| 6. Train/Val/Test     | 70/15/15 split, saved as compressed `.npz` chunks             |

## Dataset Details

- **Source:** Sentinel-1 & Sentinel-2 satellite imagery
- **Region:** Brazilian Amazon (EPSG:32722 — UTM zone 22S)
- **Classes:** Deforestation (1) vs Non-Deforestation (0)
- **Image size:** 16 patches of 2816×2816 pixels each
