import torch
import os

# --- Hardware / Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Paths Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# --- SynthMorph General Pipeline ---
TARGET_SHAPE = (160, 192, 224)

# ===========================================================
# --- sm-shapes Baseline Generator (Variant A) ---
# ===========================================================
SHAPES_NUM_LABELS   = 26           # J: tăng lên 26 (theo yêu cầu của bạn) để viền label nhỏ lại
SHAPES_NOISE_RES    = (10, 12, 14) # Low-res noise for blobby shapes
SHAPES_SVF_RES      = (10, 12, 14) # SVF warp resolution for shapes
SHAPES_SVF_STD      = 3.0          # Tăng biên độ vặn xoắn lên 3.0 để gây sai số lớn lúc đầu (tránh Loss rớt luôn xuống 0.15)


# ===========================================================
# --- Custom Generator (Variant B — Shape-Focused) ---
# ===========================================================
CUSTOM_NUM_LABELS    = 26          # J=26 (more diversity than baseline J=15)
CUSTOM_ALPHA         = 1.0         # Weight for geometric primitive scores
CUSTOM_BETA          = 0.5         # Weight for blob (low-freq noise) scores
CUSTOM_BLOB_RES      = (10, 12, 14)  # Blob noise resolution (same as baseline)

# Multi-scale SVF warping: 3 levels from global -> micro
CUSTOM_SVF_GLOBAL_RES = (16, 16, 16)  # Large-scale alignment
CUSTOM_SVF_GLOBAL_STD = 3.0
CUSTOM_SVF_LOCAL_RES  = (8,  8,  8)   # Mid-scale (local twist/fold)
CUSTOM_SVF_LOCAL_STD  = 1.5
CUSTOM_SVF_MICRO_RES  = (4,  4,  4)   # Fine-scale (inflate/deflate)
CUSTOM_SVF_MICRO_STD  = 0.5

# Primitive sizes (relative to volume, e.g. 0.05 = 5% of dim)
CUSTOM_PRIM_SIZE_MIN  = 0.04       # Min primitive radius/half-size
CUSTOM_PRIM_SIZE_MAX  = 0.25       # Max primitive radius/half-size

# Deformation augmentation priors
CUSTOM_TWIST_STD      = 1.2        # SVF std for twist deformation
CUSTOM_INFLATE_STD    = 1.0        # SVF std for inflate/deflate
CUSTOM_FOLD_STD       = 0.8        # SVF std for local folding

# ===========================================================
# --- sm-brains (Label Generator) ---
# ===========================================================
BUCKNER40_URL = "https://surfer.nmr.mgh.harvard.edu/pub/data/tutorial_data.tar.gz"

# ===========================================================
# --- Spatial Transform (used in final warp, both variants) ---
# ===========================================================
SPATIAL_SVF_RES             = (16, 16, 16)
SPATIAL_SVF_STD             = 3.0
SCALING_AND_SQUARING_STEPS  = 7

# ===========================================================
# --- Intensity Generator (Synthetic MRI) ---
# ===========================================================
INTENSITY_GMM_MEAN_RANGE = (0.2, 0.8)
INTENSITY_GMM_STD_RANGE  = (0.01, 0.1)

# ===========================================================
# --- Artifact & Augmentation ---
# ===========================================================
BLUR_STD_RANGE   = (0.5, 1.5)
BIAS_FIELD_RES   = (4, 4, 4)
BIAS_FIELD_STD   = 0.3
GAMMA_RANGE      = (0.5, 2.0)

# ===========================================================
# --- Training Hyperparameters ---
# ===========================================================
# Network
NB_FEATURES         = 64           # Local safe (8GB VRAM). Cloud: 256 (via --nb-features)
NB_FEATURES_CLOUD   = 256          # Paper standard — set via CLI --nb-features on Colab/Kaggle
INTEGRATION_STEPS   = 5

# Loss
NUM_LABELS  = 26
LAMBDA_REG  = 1.0

# Optimizer
LEARNING_RATE     = 1e-4
LEARNING_RATE_MIN = 1e-5

# Training schedule
TOTAL_ITERS = 400_000
BATCH_SIZE  = 1

# ===========================================================
# --- Logging / Checkpointing / Visualization ---
# All settable via CLI args; these are the defaults
# ===========================================================
LOG_EVERY   = 10       # Print loss every N iters
VIS_EVERY   = 20       # Save step1/2/3 artifact images every N iters
SAVE_EVERY  = 20       # Save periodic checkpoint + best model every N iters

# Cloud Drive backup (only used when --drive-dir is provided)
DRIVE_COPY_EVERY = 20  # How often to sync checkpoint to persistent storage
