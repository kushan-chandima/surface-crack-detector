"""
Configuration constants for the Surface Crack Detector.
All hyperparameters, paths, and settings are centralized here.
"""

import os

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Dataset paths
DATASET_NAME = "arunrk7/surface-crack-detection"
RAW_DATA_DIR = os.path.join(DATA_DIR, "surface-crack-detection")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Model save paths
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_resaved.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")

# ==============================================================================
# Image & Data Settings
# ==============================================================================
IMG_SIZE = (224, 224)      # MobileNetV2 native input size
IMG_SHAPE = (224, 224, 3)  # Input shape with channels
BATCH_SIZE = 32
CLASS_NAMES = ["Negative", "Positive"]  # 0 = No Crack, 1 = Crack

# Data split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==============================================================================
# Model Settings
# ==============================================================================
# Options: "mobilenetv2" (recommended) or "custom_cnn"
MODEL_TYPE = "mobilenetv2"

# MobileNetV2 Transfer Learning settings
FREEZE_BASE = True      # Freeze base model layers
DROPOUT_RATE = 0.3      # Dropout rate for regularization
DENSE_UNITS = 128       # Dense layer units in classification head

# Custom CNN settings
CNN_FILTERS = [32, 64, 128]  # Number of filters per conv block

# ==============================================================================
# Training Hyperparameters
# ==============================================================================
EPOCHS = 30
LEARNING_RATE = 1e-4     # Lower LR for transfer learning
EARLY_STOP_PATIENCE = 5  # Stop if no improvement for N epochs
REDUCE_LR_PATIENCE = 3   # Reduce LR if no improvement for N epochs
REDUCE_LR_FACTOR = 0.5   # Multiply LR by this factor

# ==============================================================================
# Data Augmentation Settings
# ==============================================================================
AUGMENTATION = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.15,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest",
}

# ==============================================================================
# Prediction Settings
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Probability threshold for crack detection

# ==============================================================================
# Visualization Settings
# ==============================================================================
FIGSIZE = (10, 8)
GRAD_CAM_ALPHA = 0.4    # Overlay transparency for Grad-CAM
