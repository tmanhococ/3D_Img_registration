import torch
import torch.nn.functional as F
import math

from src import config
from src.pipeline.sm_shapes_pipeline import SynthMorphShapesPipeline
import matplotlib.pyplot as plt

def test_pipeline():
    print(f"Device: {config.DEVICE}")
    pipeline = SynthMorphShapesPipeline(device=config.DEVICE, generator_type='baseline')
    
    m, f, s_m, s_f = pipeline.generate_pair(batch_size=1)
    
    print("--- Stats ---")
    print(f"m:   shape={m.shape}, min={m.min().item():.3f}, max={m.max().item():.3f}, mean={m.mean().item():.3f}")
    print(f"f:   shape={f.shape}, min={f.min().item():.3f}, max={f.max().item():.3f}, mean={f.mean().item():.3f}")
    print(f"s_m: shape={s_m.shape}, min={s_m.min().item()}, max={s_m.max().item()}")
    print(f"s_f: shape={s_f.shape}, min={s_f.min().item()}, max={s_f.max().item()}")
    
    # Save a coronal slice to check visually
    slice_idx = m.shape[3] // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(s_m[0, 0, :, slice_idx, :].cpu().numpy(), cmap='nipy_spectral')
    axes[0, 0].set_title("Moving Label (s_m)")
    
    axes[0, 1].imshow(s_f[0, 0, :, slice_idx, :].cpu().numpy(), cmap='nipy_spectral')
    axes[0, 1].set_title("Fixed Label (s_f)")
    
    axes[1, 0].imshow(m[0, 0, :, slice_idx, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("Moving Image (m)")
    
    axes[1, 1].imshow(f[0, 0, :, slice_idx, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title("Fixed Image (f)")
    
    plot_path = 'debug_plot.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    with torch.no_grad():
        test_pipeline()
