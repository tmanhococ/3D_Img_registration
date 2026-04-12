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
# --- sm-shapes Baseline Generator (Variant A - IEEE Matched) ---
# ===========================================================
SHAPES_NUM_LABELS   = 26           # J: 26 (from paper)
SHAPES_NOISE_RES    = (5, 6, 7)    # rp = 1:32 relative to (160, 192, 224)
SHAPES_MAX_SVF_STD  = 100.0        # bp = 100 max std for individual SVF warping

# ===========================================================
# --- Custom Generator (Variant B — Geometric Inherited) ---
# ===========================================================
CUSTOM_NUM_LABELS    = 26          
CUSTOM_ALPHA         = 1.0         # Weight for geometric primitive scores
CUSTOM_BETA          = 0.5         # Weight for blob (low-freq noise) scores
CUSTOM_BLOB_RES      = (5, 6, 7)   # Inherit standard low-res noise from baseline

# Multi-scale SVF warping geometric properties
CUSTOM_SVF_GLOBAL_RES = (16, 16, 16)  
CUSTOM_SVF_GLOBAL_STD = 3.0
CUSTOM_SVF_LOCAL_RES  = (8,  8,  8)   
CUSTOM_SVF_LOCAL_STD  = 1.5
CUSTOM_SVF_MICRO_RES  = (4,  4,  4)   
CUSTOM_SVF_MICRO_STD  = 0.5

# Primitive sizes (relative to volume, e.g. 0.05 = 5% of dim)
CUSTOM_PRIM_SIZE_MIN  = 0.08       
CUSTOM_PRIM_SIZE_MAX  = 0.45       

# Deformation augmentation priors
CUSTOM_TWIST_STD      = 1.2        
CUSTOM_INFLATE_STD    = 1.0        
CUSTOM_FOLD_STD       = 0.8        

# ===========================================================
# --- sm-brains (Label Generator) ---
# ===========================================================
BUCKNER40_URL = "https://surfer.nmr.mgh.harvard.edu/pub/data/tutorial_data.tar.gz"

# ===========================================================
# --- Spatial Transform (Fixed/Moving Pair Generation) ---
# ===========================================================
MULTI_SVF_RES               = [(20, 24, 28), (10, 12, 14), (5, 6, 7)] # rv in {1:8, 1:16, 1:32}
MULTI_SVF_MAX_STD           = 3.0  # bv = 3
SCALING_AND_SQUARING_STEPS  = 7

# ===========================================================
# --- Intensity Generator (Synthetic MRI) ---
# ===========================================================
INTENSITY_GMM_MEAN_RANGE = (25.0, 225.0)  # a_mu, b_mu
INTENSITY_GMM_STD_RANGE  = (5.0, 25.0)    # a_sigma, b_sigma

# ===========================================================
# --- Artifact & Augmentation ---
# ===========================================================
BLUR_MAX_STD     = 1.0           # b_K
BIAS_FIELD_RES   = (4, 5, 6)     # r_B = 1:40
BIAS_FIELD_MAX_STD = 0.3         # b_B
GAMMA_STD        = 0.25          # sigma_gamma

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
