"""
Evaluation Metrics for SynthMorph

Metrics:
  - dice_score: Overlap between warped moving labels and fixed labels
  - jacobian_determinant: Measures diffeomorphism quality of deformation field
    (negative Jacobian determinant values indicate folding/topology violation)
"""

import torch
import torch.nn.functional as F
import numpy as np


def dice_score(
    warped_labels: torch.Tensor,
    fixed_labels: torch.Tensor,
    num_labels: int = 26,
    eps: float = 1e-6,
) -> dict:
    """
    Compute per-class and mean Dice score.

    Args:
        warped_labels: (B, 1 or J, D, H, W) — warped moving label map
        fixed_labels:  (B, 1 or J, D, H, W) — fixed label map
        num_labels:    number of classes J
        eps:           stability epsilon

    Returns:
        dict with keys:
          'mean_dice': float, mean Dice across all labels and batch
          'per_class': np.ndarray (J,) Dice per label, averaged over batch
    """
    def to_onehot(lm: torch.Tensor) -> torch.Tensor:
        if lm.shape[1] == num_labels:
            return lm.float()
        lm_int = lm.squeeze(1).long().clamp(0, num_labels - 1)
        B, D, H, W = lm_int.shape
        oh = torch.zeros(B, num_labels, D, H, W, device=lm.device, dtype=torch.float32)
        oh.scatter_(1, lm_int.unsqueeze(1), 1.0)
        return oh

    pred = to_onehot(warped_labels)  # (B, J, D, H, W)
    true = to_onehot(fixed_labels)   # (B, J, D, H, W)

    B, J = pred.shape[:2]
    pred_flat = pred.view(B, J, -1)  # (B, J, N)
    true_flat = true.view(B, J, -1)

    inter = (pred_flat * true_flat).sum(dim=-1)       # (B, J)
    denom = pred_flat.sum(dim=-1) + true_flat.sum(dim=-1)  # (B, J)
    dice  = (2.0 * inter + eps) / (denom + eps)       # (B, J)

    per_class = dice.mean(dim=0).detach().cpu().numpy()  # (J,)
    mean_dice = float(per_class.mean())

    return {'mean_dice': mean_dice, 'per_class': per_class}


def jacobian_determinant(phi: torch.Tensor) -> dict:
    """
    Compute the Jacobian determinant of a 3D displacement field.

    Negative values indicate topology-violating folds (non-diffeomorphic).

    Args:
        phi: (B, 3, D, H, W) displacement field (not velocity field)

    Returns:
        dict with keys:
          'mean_det':      float, mean Jacobian determinant (should be ~1)
          'std_det':       float, std of Jacobian determinant
          'pct_negative':  float, percentage of voxels with det < 0 (ideal: 0%)
          'min_det':       float, minimum determinant value
    """
    B = phi.shape[0]
    results = {'mean_det': [], 'std_det': [], 'pct_negative': [], 'min_det': []}

    for b in range(B):
        # u = phi (displacement), so x_deformed = x + u
        # J(x) = I + ∇u  →  det(J) = det(I + ∇u)
        u = phi[b]  # (3, D, H, W)

        D_dim, H_dim, W_dim = u.shape[1], u.shape[2], u.shape[3]

        # Compute spatial gradients via finite differences
        # du0/dx0, du0/dx1, du0/dx2, etc.
        # Shape after diff: one less in that axis
        # We'll pad to keep shape consistent

        def grad_along(t, axis):
            """Forward difference with zero-padding at end."""
            d = torch.diff(t, dim=axis)
            pad_shape = list(t.shape)
            pad_shape[axis] = 1
            pad = torch.zeros(pad_shape, device=t.device, dtype=t.dtype)
            return torch.cat([d, pad], dim=axis)

        # J = I + du  where du is the gradient of displacement
        # Diagonal entries of du:
        du0_dx0 = grad_along(u[0], 0)   # d(u_z)/dz
        du1_dx1 = grad_along(u[1], 1)   # d(u_y)/dy
        du2_dx2 = grad_along(u[2], 2)   # d(u_x)/dx

        # Off-diagonal (for full 3x3 Jacobian det)
        du0_dx1 = grad_along(u[0], 1)
        du0_dx2 = grad_along(u[0], 2)
        du1_dx0 = grad_along(u[1], 0)
        du1_dx2 = grad_along(u[1], 2)
        du2_dx0 = grad_along(u[2], 0)
        du2_dx1 = grad_along(u[2], 1)

        # Jacobian matrix J at each voxel: (3, 3, D, H, W)
        # J = I + ∇u
        j00 = 1.0 + du0_dx0
        j11 = 1.0 + du1_dx1
        j22 = 1.0 + du2_dx2
        j01, j02 = du0_dx1, du0_dx2
        j10, j12 = du1_dx0, du1_dx2
        j20, j21 = du2_dx0, du2_dx1

        # 3x3 determinant formula
        det = (j00 * (j11 * j22 - j12 * j21)
             - j01 * (j10 * j22 - j12 * j20)
             + j02 * (j10 * j21 - j11 * j20))

        det_np = det.detach().cpu().numpy().flatten()
        results['mean_det'].append(float(np.mean(det_np)))
        results['std_det'].append(float(np.std(det_np)))
        results['pct_negative'].append(float(np.mean(det_np < 0) * 100))
        results['min_det'].append(float(np.min(det_np)))

    return {
        'mean_det':     float(np.mean(results['mean_det'])),
        'std_det':      float(np.mean(results['std_det'])),
        'pct_negative': float(np.mean(results['pct_negative'])),
        'min_det':      float(np.mean(results['min_det'])),
    }
