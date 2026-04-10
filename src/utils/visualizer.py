"""
Visualization Utility for SynthMorph Pipeline

Saves 2D slice snapshots at 3 checkpoints in the pipeline:
  - Step 1: After label map generation (s_m, s_f colored by label)
  - Step 2: After synthetic MRI generation (m, f grayscale intensity)
  - Step 3: After registration network (6-panel: fixed/moving/warped images + labels + deform grid)

All images are saved as .png under runs/vis/ and also as .npy for numerical inspection.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend, no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch


def _get_mid_slice(tensor: torch.Tensor) -> np.ndarray:
    """
    Extract the middle axial (z-axis) slice from a 3D volume tensor.

    Args:
        tensor: (B, C, D, H, W) or (B, 1, D, H, W)
    Returns:
        2D numpy array of shape (H, W), first batch, first channel
    """
    arr = tensor[0, 0].detach().cpu().float().numpy()   # (D, H, W)
    mid = arr.shape[0] // 2
    return arr[mid]   # (H, W)


def _label_slice_to_rgb(label_tensor: torch.Tensor, num_labels: int = 26) -> np.ndarray:
    """
    Convert a label map slice to an RGB image using a discrete colormap.

    Args:
        label_tensor: (B, 1 or J, D, H, W) — either integer single-channel or one-hot
        num_labels: number of classes
    Returns:
        RGB numpy array (H, W, 3), float [0,1]
    """
    if label_tensor.shape[1] == 1:
        # Integer label map
        arr = label_tensor[0, 0].detach().cpu().float().numpy()  # (D, H, W)
    else:
        # One-hot: argmax over label axis
        arr = label_tensor[0].detach().cpu().float().argmax(dim=0).numpy()  # (D, H, W)

    mid = arr.shape[0] // 2
    slice_2d = arr[mid]  # (H, W)

    cmap = plt.get_cmap('tab20', num_labels)
    norm = mcolors.BoundaryNorm(boundaries=range(num_labels + 1), ncolors=num_labels)
    rgba = cmap(norm(slice_2d.astype(int)))   # (H, W, 4)
    return rgba[:, :, :3]   # drop alpha


def _deform_grid_slice(phi: torch.Tensor, step: int = 8) -> np.ndarray:
    """
    Visualize a 2D deformation grid (middle axial slice of phi).

    Args:
        phi: (B, 3, D, H, W) displacement field
        step: grid line spacing in pixels
    Returns:
        RGB image (H, W, 3)
    """
    phi_arr = phi[0].detach().cpu().numpy()   # (3, D, H, W)
    mid = phi_arr.shape[1] // 2

    # Horizontal and vertical components at mid slice
    dy = phi_arr[1, mid]   # (H, W) — displacement in H direction
    dx = phi_arr[2, mid]   # (H, W) — displacement in W direction
    H, W = dy.shape

    # Create identity grid
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Deformed grid positions
    def_y = grid_y + dy
    def_x = grid_x + dx

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('black')

    for r in range(0, H, step):
        ax.plot(def_x[r, :], def_y[r, :], 'c-', linewidth=0.5, alpha=0.7)
    for c in range(0, W, step):
        ax.plot(def_x[:, c], def_y[:, c], 'c-', linewidth=0.5, alpha=0.7)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    buf = buf.reshape(height, width, 4)
    plt.close(fig)
    return buf[:, :, :3].astype(np.float32) / 255.0


class Visualizer:
    """
    Central visualization manager for SynthMorph pipeline.

    Usage:
        viz = Visualizer(save_dir='runs/vis', save_every=1000)

        # Inside training loop:
        viz.save_step1(s_m, s_f, iteration)
        viz.save_step2(m, f, iteration)
        viz.save_step3(m, f, warped_m, s_f, warped_sm, phi, iteration)

    All saves are only executed every `save_every` iterations.
    """

    def __init__(self, save_dir: str = 'runs/vis', save_every: int = 1000):
        self.save_dir = save_dir
        self.save_every = save_every
        os.makedirs(save_dir, exist_ok=True)

    def _should_save(self, iteration: int) -> bool:
        return (iteration % self.save_every) == 0

    def _save_npy(self, arr: np.ndarray, path: str):
        np.save(path, arr)

    # ------------------------------------------------------------------
    # STEP 1: Label map visualization
    # ------------------------------------------------------------------
    def save_step1(self, s_m: torch.Tensor, s_f: torch.Tensor, iteration: int,
                   num_labels: int = 26):
        """
        Save Step 1 visualization: moving and fixed label maps (colored).

        Args:
            s_m: Moving label map (B, 1 or J, D, H, W)
            s_f: Fixed  label map (B, 1 or J, D, H, W)
            iteration: current training iteration
            num_labels: total number of label classes
        """
        if not self._should_save(iteration):
            return

        rgb_sm = _label_slice_to_rgb(s_m, num_labels)
        rgb_sf = _label_slice_to_rgb(s_f, num_labels)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'Step 1: Label Maps — Iter {iteration}', fontsize=13)

        axes[0].imshow(rgb_sm)
        axes[0].set_title('Moving Label Map $s_m$')
        axes[0].axis('off')

        axes[1].imshow(rgb_sf)
        axes[1].set_title('Fixed Label Map $s_f$')
        axes[1].axis('off')

        plt.tight_layout()
        fname = os.path.join(self.save_dir, f'step1_labels_iter{iteration:06d}.png')
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[Viz] Saved Step 1 → {fname}')

        # Also save raw numpy for numerical inspection
        self._save_npy(rgb_sm, fname.replace('.png', '_sm.npy'))
        self._save_npy(rgb_sf, fname.replace('.png', '_sf.npy'))

    # ------------------------------------------------------------------
    # STEP 2: Synthetic MRI visualization
    # ------------------------------------------------------------------
    def save_step2(self, m: torch.Tensor, f: torch.Tensor, iteration: int):
        """
        Save Step 2 visualization: synthetic moving and fixed MR images.

        Args:
            m: Moving image (B, 1, D, H, W)
            f: Fixed  image (B, 1, D, H, W)
            iteration: current training iteration
        """
        if not self._should_save(iteration):
            return

        slice_m = _get_mid_slice(m)
        slice_f = _get_mid_slice(f)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'Step 2: Synthetic MRI — Iter {iteration}', fontsize=13)

        axes[0].imshow(slice_m, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Moving Image $m$')
        axes[0].axis('off')

        axes[1].imshow(slice_f, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Fixed Image $f$')
        axes[1].axis('off')

        plt.tight_layout()
        fname = os.path.join(self.save_dir, f'step2_mri_iter{iteration:06d}.png')
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[Viz] Saved Step 2 → {fname}')

        self._save_npy(slice_m, fname.replace('.png', '_m.npy'))
        self._save_npy(slice_f, fname.replace('.png', '_f.npy'))

    # ------------------------------------------------------------------
    # STEP 3: Registration result visualization (6-panel grid)
    # ------------------------------------------------------------------
    def save_step3(
        self,
        m: torch.Tensor,
        f: torch.Tensor,
        warped_m: torch.Tensor,
        s_f: torch.Tensor,
        warped_sm: torch.Tensor,
        phi: torch.Tensor,
        iteration: int,
        num_labels: int = 26,
    ):
        """
        Save Step 3 visualization: 6-panel registration result.

        Panels: Fixed | Moving | Warped Moving
                Fixed Label | Warped Label | Deformation Grid

        Args:
            m:         Moving image          (B, 1, D, H, W)
            f:         Fixed  image          (B, 1, D, H, W)
            warped_m:  m warped by phi       (B, 1, D, H, W)
            s_f:       Fixed label map       (B, 1 or J, D, H, W)
            warped_sm: s_m warped by phi     (B, 1 or J, D, H, W)
            phi:       Displacement field    (B, 3, D, H, W)
            iteration: current training iteration
            num_labels: number of label classes
        """
        if not self._should_save(iteration):
            return

        slice_f  = _get_mid_slice(f)
        slice_m  = _get_mid_slice(m)
        slice_wm = _get_mid_slice(warped_m)
        rgb_sf   = _label_slice_to_rgb(s_f, num_labels)
        rgb_wsm  = _label_slice_to_rgb(warped_sm, num_labels)
        grid_img = _deform_grid_slice(phi)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(f'Step 3: Registration — Iter {iteration}', fontsize=14)

        # Row 0: Intensity images
        axes[0, 0].imshow(slice_f,  cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Fixed Image $f$')

        axes[0, 1].imshow(slice_m,  cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Moving Image $m$')

        axes[0, 2].imshow(slice_wm, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('Warped Moving $m \\circ \\phi$')

        # Row 1: Labels + deformation grid
        axes[1, 0].imshow(rgb_sf)
        axes[1, 0].set_title('Fixed Label $s_f$')

        axes[1, 1].imshow(rgb_wsm)
        axes[1, 1].set_title('Warped Label $s_m \\circ \\phi$')

        axes[1, 2].imshow(grid_img)
        axes[1, 2].set_title('Deformation Grid $\\phi$')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        fname = os.path.join(self.save_dir, f'step3_registration_iter{iteration:06d}.png')
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[Viz] Saved Step 3 → {fname}')

        # Save raw slices
        self._save_npy(slice_wm,  fname.replace('.png', '_warped_m.npy'))
        self._save_npy(rgb_wsm,   fname.replace('.png', '_warped_label.npy'))
