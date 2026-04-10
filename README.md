# SynthMorph Generative Pipeline

A PyTorch-based, modular, GPU/CPU agile implementation of the SynthMorph 2022 generative model, aimed at synthesizing diverse 3D anatomical labels (`sm-shapes`), or applying perturbations to real dataset segmentation maps (`sm-brains`) and producing augmented realistic Brain MRIs paired with spatial deformation fields. This project enables unsupervised deformation model training dynamically natively inside PyTorch.

## What's New in Phase 2: Cloud Scaling & Shape-focused Variant B 🚀
- **Streaming Argmax Custom Generator (Variant B):** We introduced a highly advanced synthetic label generator combining geometric primitives (Sphere, Ellipsoid, Cylinder, Cuboid) with multi-scale global/local/micro perturbations and complex deformation topologies (Twist, Inflate, Deflate, Fold). Designed for extremely low VRAM usage via **streaming CPU-GPU argmax** calculations. (Peak GPU VRAM usage dropped from 17GB to < 400MB for label synthesis!).
- **Cloud-Ready Training CLI:** Unified `train_sm_shapes.py` entry point ready for Google Colab & Kaggle out of the box with safe encoding parameters, `nb_features` adaptability, and explicit AMP toggles.
- **Persistent Cloud Checkpointing:** Designed continuous Google Drive & Cloud syncing, saving Best models and interval models continuously to overcome the 12-hour timeout thresholds of cloud platforms.

## Structure

```
.
├── src/
│   ├── config.py                       # Main hyperparameters (Target Shapes, SVF Scales)
│   ├── data/
│   ├── generators/
│   │   ├── labels/                     
│   │   │   ├── sm_shapes_labels.py         # Variant A: Standard Blob & SVF labels
│   │   │   ├── custom_shapes_labels.py     # Variant B: Geometric Primitives + Multi SVF
│   │   │   └── sm_brains_labels.py
│   │   ├── spatial/                    # Diffeomorphic transform & SVF Integrators 
│   │   └── intensity/                  # GMM Rendering & Intensity Augmentations
│   ├── pipeline/
│   ├── training/                       # Checkpointing, OOM Handling & Core Loop Logic
│   └── utils/
├── notebooks/                          
│   ├── colab_train_variant_b.ipynb     # 🔥 Ready-to-go Colab Notebook (Variant B)
│   ├── kaggle_train_sm_shapes.ipynb    # 🔥 Ready-to-go Kaggle Notebook (Variant A)
│   └── debug_pipeline.ipynb            
├── train_sm_shapes.py                  # Entry Point for Training
└── requirements.txt                    
```

## Running Training on Cloud (Google Colab & Kaggle)

You can scale training seamlessly using our pre-configured Jupyter notebooks available in the `notebooks/` directory.

### Quick Run Setup
1. **Google Colab:** Open `colab_train_variant_b.ipynb`. We have prepared Google Drive mounting automatically so checkpoints (`.pth`) sync seamlessly preventing loss when runtime disconnects.
2. **Kaggle:** Open `kaggle_train_sm_shapes.ipynb`, built dynamically to archive results and outputs nicely in Kaggle's working directory.

### Usage Example
```bash
# Dry-run with Variant B Custom features
python train_sm_shapes.py --generator-type custom --iters 20 --nb-features 64

# Cloud Training Variant B (Full scale)
python train_sm_shapes.py --generator-type custom --iters 400000 --nb-features 256 \
                          --runs-dir /content/SynthMorph/runs \
                          --drive-dir /content/drive/MyDrive/SynthMorph_runs
```
*Note: We employ an `OOMHandler` that automatically steps down the `nb_features` parameter in architecture sizes if native VRAM checks fail.*

## Visualisation

Training will actively write synthetic steps (`labels -> synthetic MRIs -> Warped pairs`) continuously onto the local disk inside the `runs/.../vis` directory every N iterations. Check inside the notebooks to preview them directly! 
