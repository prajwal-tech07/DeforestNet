"""
DeforestNet - Project Configuration
Central configuration for all paths, parameters, and hyperparameters.
"""

import os

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

# Dataset paths (original .tif files)
DATASET_DIR = os.path.join(WORKSPACE_ROOT, "dataset")
CLOUD_FREE_DIR = os.path.join(DATASET_DIR, "1_CLOUD_FREE_DATASET")
CLOUDY_DIR = os.path.join(DATASET_DIR, "2_CLOUDY_DATASET")
MASKS_DIR = os.path.join(DATASET_DIR, "3_TRAINING_MASKS")

# Sentinel sub-paths
CLOUD_FREE_S1 = os.path.join(CLOUD_FREE_DIR, "1_SENTINEL1")
CLOUD_FREE_S2 = os.path.join(CLOUD_FREE_DIR, "2_SENTINEL2")
CLOUDY_S1 = os.path.join(CLOUDY_DIR, "1_SENTINEL1")
CLOUDY_S2 = os.path.join(CLOUDY_DIR, "2_SENTINEL2")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "preprocessed")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# ============================================================
# DATASET PARAMETERS
# ============================================================
# Coordinate Reference System
CRS = "EPSG:32722"  # WGS 84 / UTM zone 22S (Brazilian Amazon)

# Sentinel-1 bands: VV (band 1), VH (band 2)
S1_BANDS = ["VV", "VH"]
S1_NUM_BANDS = 2

# Sentinel-2 bands: Blue (B2), Green (B3), Red (B4), NIR (B8)
S2_BANDS = ["B2_Blue", "B3_Green", "B4_Red", "B8_NIR"]
S2_NUM_BANDS = 4

# Mask classes
MASK_CLASSES = {
    0: "NoData",
    1: "Deforestation",
    2: "Non-Deforestation"
}
NUM_CLASSES = 2  # Binary: Deforestation vs Non-Deforestation (ignoring NoData)

# ============================================================
# PREPROCESSING PARAMETERS
# ============================================================
# Image patch size (the dataset already has 16 patches of this size)
FULL_IMAGE_SIZE = 11264
PATCH_SIZE = 256  # Size of training patches to extract
PATCH_STRIDE = 128  # Stride for overlapping patches (50% overlap)

# Noise removal
MEDIAN_FILTER_SIZE = 3  # Kernel size for median filtering (Sentinel-1 speckle)
GAUSSIAN_SIGMA = 1.0  # Sigma for Gaussian smoothing

# Normalization approach
# Options: "minmax", "standardize", "percentile"
NORMALIZATION_METHOD = "percentile"
PERCENTILE_LOW = 2   # Lower percentile for clipping
PERCENTILE_HIGH = 98  # Upper percentile for clipping

# ============================================================
# DATA SPLIT
# ============================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================
# NDVI PARAMETERS
# ============================================================
# NDVI = (NIR - Red) / (NIR + Red)
# For Sentinel-2: NIR = Band 4 (B8), Red = Band 3 (B4)
NDVI_NIR_BAND_INDEX = 3  # 0-indexed: B8_NIR
NDVI_RED_BAND_INDEX = 2  # 0-indexed: B4_Red

# ============================================================
# DATALOADER PARAMETERS
# ============================================================
BATCH_SIZE = 16
NUM_WORKERS = 0  # Use 0 on Windows; increase on Linux
PIN_MEMORY = True
IN_CHANNELS = 11  # Total feature bands (6 sensor + 5 derived indices)

# ============================================================
# MODEL PARAMETERS
# ============================================================
MODEL_NAME = "UNetResNet34"
DROPOUT_P = 0.2  # Dropout before final classification head
ENCODER_NAME = "resnet34"

# ============================================================
# LOSS FUNCTION PARAMETERS
# ============================================================
LOSS_TYPE = "combined"  # Options: "ce", "dice", "focal", "combined"
# Class weights: [non-deforestation, deforestation]
# Derived from training set distribution (73.1% / 26.9%)
CLASS_WEIGHTS = [0.684, 1.857]
# Combined loss component weights
DICE_LOSS_WEIGHT = 0.5
CE_LOSS_WEIGHT = 0.5
FOCAL_LOSS_WEIGHT = 0.0  # Set > 0 to enable focal loss component
FOCAL_GAMMA = 2.0  # Focusing parameter for focal loss
DICE_SMOOTH = 1.0  # Smoothing factor for dice loss

# ============================================================
# METRIC PARAMETERS
# ============================================================
CLASS_NAMES = ["Non-Deforest", "Deforest"]
