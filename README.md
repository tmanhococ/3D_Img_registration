# SynthMorph Generative Pipeline

A PyTorch-based, modular, GPU/CPU agile implementation of the SynthMorph 2022 generative model, aimed at synthesizing diverse 3D anatomical labels (`sm-shapes`), or applying perturbations to real dataset segmentation maps (`sm-brains`) and producing augmented realistic Brain MRIs paired with spatial deformation fields. This project enables unsupervised deformation model training dynamically natively inside PyTorch.

## Structure

```
.
├── src/
│   ├── config.py                       # Main application hyperparameters (Resolutions, scales)
│   ├── data/
│   │   ├── buckner_downloader.py       # Utility to wget freeSurfer tutorial dataset.
│   │   ├── loader.py                   # Loading structural maps via nibabel.
│   │   └── preprocess.py               # Nearest neighbor cropping and reshaping.
│   ├── generators/
│   │   ├── labels/                     # Label map (Shape or Brains) generation base methods
│   │   │   ├── sm_shapes_labels.py
│   │   │   └── sm_brains_labels.py
│   │   ├── spatial/                    # Diffeomorphic transform modules
│   │   │   ├── svf_generator.py        # Produce Stationary velocity fields
│   │   │   ├── scaling_squaring.py     # Integrates velocity fields
│   │   │   └── warper.py               # F.grid_sample utilities
│   │   └── intensity/                  # Synthetic rendering strategies
│   │       ├── gmm_sampler.py          # Independent label intensities
│   │       ├── blur_pve.py             # Anisotropic Partial Volume effect blurs
│   │       ├── bias_field.py           # Multiplicative inhomogeneous noise
│   │       └── augmentation.py         # Min/Max normalization & gamma correction
│   └── pipeline/
│       ├── sm_shapes_pipeline.py       # Entry point for purely synthetic shapes
│       └── sm_brains_pipeline.py       # Entry point leveraging Buckner40 data
├── notebooks/                          # Jupyter Notebook for visualization and debugging
└── requirements.txt                    # Project Dependencies
```

## Quickstart Configuration

The pipeline inherently supports native batching and Device Agnosticism.
Check out `src/config.py` to modify settings:
- `DEVICE` is automatically configured based on `torch.cuda.is_available()` resolving cleanly whether utilizing Colab GPUs or Local CPUs. 
- You can tune SVF standard deviation (`SHAPES_SVF_STD` & `SPATIAL_SVF_STD`) here.

## Running the Data Downloader

For `sm-brains`, the pipeline automatically checks for Buckner40 data upon request, downloading the standard `tutorial_data.tar.gz` and extracting recursively for `aseg.mgz` automatically.
Should you wish to trigger it manually:
```bash
python -m src.data.buckner_downloader
```

## Generating Images

Example snippet for generating a completely synthetic (`sm-shapes`) matching image pair ready for registration network training:

```python
from src.pipeline.sm_shapes_pipeline import SynthMorphShapesPipeline

# Initialize Pipeline (handles Config device dynamically)
pipeline = SynthMorphShapesPipeline()

# Produce pairs (Batch Support natively integrated)
moving_img, fixed_img, moving_label, fixed_label = pipeline.generate_pair(batch_size=2)
```

## Hardware Note
Because this code relies fully on `torch` logic, operations that historically relied on `scipy` limits like Anisotropic filters and Coordinate Grid interpolators now occur safely across the chosen PyTorch GPU/CPU architectures in full tensor arrays.
