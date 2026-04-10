"""
Loss Functions for SynthMorph Training (Step 4)

NOTE: vxm.nn.losses.Dice and vxm.nn.losses.Grad in the cloned VoxelMorph dev
branch are deprecated stubs (raise NotImplementedError). We implement both
losses from scratch in pure PyTorch.

Losses implemented:
  1. SoftDiceLoss  — geometric overlap loss on label maps (L_dis)
  2. GradLoss      — gradient smoothness regularization on displacement field (L_reg)

Combined loss: L = L_dis + lambda * L_reg  (lambda = 1.0 per paper)

Reference:
  Hoffmann et al., "SynthMorph: Learning Contrast-Invariant Registration
  without Acquired Images", IEEE TMI 2022. Eq. 4-6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss over one-hot encoded label maps.

    Measures geometric overlap between the warped moving label map
    (s_m ∘ phi) and the fixed label map (s_f).

    Formula (per paper Eq. 4-5):
        L_dis = -2 * sum_j( |(s_m_j ∘ phi) · s_f_j| ) /
                       sum_j( |(s_m_j ∘ phi)| + |s_f_j| )

    Args:
        eps (float): Small epsilon to avoid division by zero.

    IMPORTANT: Operates only on label maps — NEVER on intensity images m, f.
    This ensures contrast-agnostic registration.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, warped_labels: torch.Tensor, fixed_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            warped_labels: s_m warped by phi, shape (B, J, D, H, W), float, soft/one-hot
            fixed_labels:  s_f, same shape, float, one-hot encoded
        Returns:
            Scalar loss (mean over batch and labels).
        """
        # Flatten spatial dims: (B, J, N) where N = D*H*W
        B, J = warped_labels.shape[:2]
        pred = warped_labels.view(B, J, -1)   # (B, J, N)
        true = fixed_labels.view(B, J, -1)    # (B, J, N)

        # Numerator: 2 * |pred · true|  per label channel
        intersection = (pred * true).sum(dim=-1)          # (B, J)

        # Denominator: |pred| + |true| per label channel
        denom = pred.sum(dim=-1) + true.sum(dim=-1)       # (B, J)

        # Dice per label, averaged
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)  # (B, J)

        # Return negative (we minimize)
        return 1.0 - dice.mean()


class GradLoss(nn.Module):
    """
    Gradient Regularization Loss on displacement field u (phi = Id + u).

    Penalizes large spatial gradients of the displacement field to enforce
    smooth, physically plausible deformations.

    Formula (per paper):
        L_reg = (1/2) * || ∇u ||^2

    Args:
        penalty (str): 'l1' or 'l2' (paper uses l2).
        loss_mult (float): Optional multiplier (set to voxel spacing if needed).
    """

    def __init__(self, penalty: str = 'l2', loss_mult: float = 1.0):
        super().__init__()
        assert penalty in ('l1', 'l2'), f"penalty must be 'l1' or 'l2', got '{penalty}'"
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi: Displacement field, shape (B, 3, D, H, W)
        Returns:
            Scalar regularization loss.
        """
        # Compute finite differences along each spatial axis
        # phi shape: (B, ndim, D, H, W)
        ndim = phi.shape[1]
        total = torch.tensor(0.0, device=phi.device)

        # Axes: D=2, H=3, W=4
        for ax in range(2, 2 + ndim):
            # Forward difference along this spatial axis
            d = torch.diff(phi, dim=ax)  # (B, ndim, ...) with one less in axis `ax`
            if self.penalty == 'l1':
                total = total + d.abs().mean()
            else:
                total = total + (d ** 2).mean()

        return total * self.loss_mult / ndim


class SynthMorphLoss(nn.Module):
    """
    Combined SynthMorph loss: L = L_dis + lambda * L_reg

    This class wraps SoftDiceLoss + GradLoss and handles:
      - One-hot encoding of integer label maps
      - Computing the combined loss scalar

    Args:
        num_labels (int): Number of label classes J (paper uses 26).
        lambda_reg (float): Weighting for regularization term (paper: 1.0).
        dice_eps (float): Epsilon for Soft Dice stability.
    """

    def __init__(
        self,
        num_labels: int = 26,
        lambda_reg: float = 1.0,
        dice_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.lambda_reg = lambda_reg
        self.dice_loss = SoftDiceLoss(eps=dice_eps)
        self.grad_loss = GradLoss(penalty='l2', loss_mult=1.0)

    def _to_onehot(self, label_map: torch.Tensor) -> torch.Tensor:
        """
        Convert integer label map to one-hot float tensor.

        Args:
            label_map: (B, 1, D, H, W) integer labels OR (B, J, D, H, W) already one-hot
        Returns:
            one_hot: (B, J, D, H, W) float
        """
        if label_map.shape[1] == self.num_labels:
            # Already one-hot (e.g. sm-shapes outputs soft blobs)
            return label_map.float()

        # Convert integer map: squeeze channel dim → long
        lm = label_map.squeeze(1).long()  # (B, D, H, W)
        B, D, H, W = lm.shape
        one_hot = torch.zeros(B, self.num_labels, D, H, W,
                              device=label_map.device, dtype=torch.float32)
        # Clamp labels that exceed num_labels (treat as background 0)
        lm = lm.clamp(0, self.num_labels - 1)
        one_hot.scatter_(1, lm.unsqueeze(1), 1.0)
        return one_hot

    def forward(
        self,
        phi: torch.Tensor,
        warped_sm: torch.Tensor,
        sf: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            phi:       Displacement field from network, (B, 3, D, H, W)
            warped_sm: s_m already warped by phi, (B, J or 1, D, H, W)
            sf:        Fixed label map s_f,        (B, J or 1, D, H, W)

        Returns:
            total_loss (scalar tensor), loss_dict (dict with individual components)
        """
        # Encode labels to one-hot if necessary
        warped_onehot = self._to_onehot(warped_sm)
        sf_onehot     = self._to_onehot(sf)

        # Geometric loss (on labels only — no intensity info)
        l_dis = self.dice_loss(warped_onehot, sf_onehot)

        # Smoothness regularization (on displacement field)
        l_reg = self.grad_loss(phi)

        # Combined
        total = l_dis + self.lambda_reg * l_reg

        return total, {
            'loss_total': total.item(),
            'loss_dice':  l_dis.item(),
            'loss_grad':  l_reg.item(),
        }
