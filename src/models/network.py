"""
SynthMorph Registration Network — Pure PyTorch Implementation (Step 3)

Standalone 3D U-Net that predicts a Stationary Velocity Field (SVF),
integrated via Scaling & Squaring to produce a diffeomorphic deformation
field phi_theta.

NO dependency on VoxelMorph or Neurite. All components implemented
in pure PyTorch to avoid keras/tensorflow compatibility issues.

Based on: Hoffmann et al., "SynthMorph: Learning Contrast-Invariant
Registration without Acquired Images", IEEE TMI 2022.

Architecture (per paper):
  - Encoder: 4 × Conv3D(stride=2) + LeakyReLU(0.2)
  - Decoder: 3 × Conv3D(stride=1) + Upsample + Skip connection
  - Output: 3-channel SVF (x, y, z velocity components)
  - Integration: Scaling & Squaring (5 steps) → diffeomorphic phi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Core Building Blocks
# -----------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """Single Conv3D → LeakyReLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SpatialTransformerPure(nn.Module):
    """
    3D Spatial Transformer (warps a volume by a displacement field).

    Pure PyTorch implementation using F.grid_sample.

    Args:
        mode: 'bilinear' for intensity images, 'nearest' for label maps
    """

    def __init__(self, mode: str = 'bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, vol: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol: (B, C, D, H, W) volume to warp
            phi: (B, 3, D, H, W) displacement field

        Returns:
            Warped volume (B, C, D, H, W)
        """
        B, C, D, H, W = vol.shape

        # Build meshgrid identity grid
        grid_d = torch.arange(D, dtype=torch.float32, device=vol.device)
        grid_h = torch.arange(H, dtype=torch.float32, device=vol.device)
        grid_w = torch.arange(W, dtype=torch.float32, device=vol.device)
        # shape: (D, H, W) each
        gd, gh, gw = torch.meshgrid(grid_d, grid_h, grid_w, indexing='ij')

        # Stack to (3, D, H, W) and add batch dim → (1, 3, D, H, W)
        identity = torch.stack([gd, gh, gw], dim=0).unsqueeze(0)  # (1, 3, D, H, W)

        # Add displacement to identity
        new_coords = identity + phi    # (B, 3, D, H, W)

        # Normalize to [-1, 1] for grid_sample
        new_d = 2.0 * new_coords[:, 0] / max(D - 1, 1) - 1.0
        new_h = 2.0 * new_coords[:, 1] / max(H - 1, 1) - 1.0
        new_w = 2.0 * new_coords[:, 2] / max(W - 1, 1) - 1.0

        # grid_sample expects (B, D, H, W, 3) with order (W, H, D) = (x, y, z)
        grid = torch.stack([new_w, new_h, new_d], dim=-1)  # (B, D, H, W, 3)

        mode = self.mode
        # grid_sample uses 'bilinear' for trilinear in 3D
        if mode == 'bilinear':
            interp_mode = 'bilinear'
        elif mode == 'nearest':
            interp_mode = 'nearest'
        else:
            interp_mode = mode

        return F.grid_sample(
            vol.float(),
            grid,
            mode=interp_mode,
            padding_mode='border',
            align_corners=True
        ).to(vol.dtype)


class IntegrateVelocityField(nn.Module):
    """
    Scaling and Squaring integration of a Stationary Velocity Field (SVF).

    Converts SVF v → diffeomorphic displacement field phi via:
        phi = exp(v) ≈ (v / 2^T) composed 2^T times

    Args:
        steps: number of squaring steps (paper: 5)
    """

    def __init__(self, steps: int = 5):
        super().__init__()
        self.steps = steps
        self.transformer = SpatialTransformerPure(mode='bilinear')

    def forward(self, svf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            svf: (B, 3, D, H, W) stationary velocity field
        Returns:
            phi: (B, 3, D, H, W) diffeomorphic displacement field
        """
        # Scale down
        phi = svf / (2.0 ** self.steps)

        # Squaring loop: phi = phi ∘ phi
        for _ in range(self.steps):
            phi = phi + self.transformer(phi, phi)

        return phi


# -----------------------------------------------------------------------
# U-Net Architecture
# -----------------------------------------------------------------------

class SynthMorphUNet(nn.Module):
    """
    3D U-Net Registration Network for SynthMorph.

    Takes concatenated (moving m, fixed f) images and predicts a
    diffeomorphic displacement field phi_theta = integrate(SVF).

    Args:
        nb_features: feature maps per layer (paper: 256; 8GB GPU safe: 64)
        integration_steps: scaling-and-squaring steps (paper: 5)
        in_channels: 2 (one for m, one for f, concatenated)
    """

    def __init__(
        self,
        nb_features: int = 64,
        integration_steps: int = 5,
        in_channels: int = 2,
    ):
        super().__init__()
        self.nb_features = nb_features
        self.integration_steps = integration_steps
        n = nb_features

        # ---- Encoder (4 levels, stride-2 downsampling) ----
        self.enc1 = ConvBlock3D(in_channels, n, stride=2)   # → n × D/2
        self.enc2 = ConvBlock3D(n, n, stride=2)              # → n × D/4
        self.enc3 = ConvBlock3D(n, n, stride=2)              # → n × D/8
        self.enc4 = ConvBlock3D(n, n, stride=2)              # → n × D/16

        # ---- Decoder (3 levels, stride-1 + upsample + skip) ----
        # Each decoder block input = upsampled features + skip connection features
        self.dec3 = ConvBlock3D(n + n, n, stride=1)          # skip from enc3
        self.dec2 = ConvBlock3D(n + n, n, stride=1)          # skip from enc2
        self.dec1 = ConvBlock3D(n + n, n, stride=1)          # skip from enc1

        # ---- SVF prediction head (at half resolution, 3 output channels) ----
        # Extra refinement conv at half resolution before SVF prediction
        self.pre_svf = nn.Sequential(
            ConvBlock3D(n, n, stride=1),
            ConvBlock3D(n, n, stride=1),
        )
        self.flow_conv = nn.Conv3d(n, 3, kernel_size=3, padding=1)

        # Initialize flow layer with small weights (paper convention)
        nn.init.normal_(self.flow_conv.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_conv.bias)

        # ---- Upsample to full resolution ----
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # ---- SVF Integration ----
        self.integrator = IntegrateVelocityField(steps=integration_steps)

        # ---- Final SpatialTransformer (for warping, exported separately) ----
        self._img_warper   = SpatialTransformerPure(mode='bilinear')
        self._label_warper = SpatialTransformerPure(mode='nearest')

    def forward(self, m: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            m: Moving image (B, 1, D, H, W)
            f: Fixed  image (B, 1, D, H, W)

        Returns:
            phi: Diffeomorphic displacement field (B, 3, D, H, W)
                 at full resolution, ready to warp m toward f.
        """
        # Concatenate inputs along channel dim
        x = torch.cat([m, f], dim=1)   # (B, 2, D, H, W)

        # ---- Encoder ----
        e1 = self.enc1(x)    # (B, n, D/2, H/2, W/2)
        e2 = self.enc2(e1)   # (B, n, D/4, ...)
        e3 = self.enc3(e2)   # (B, n, D/8, ...)
        e4 = self.enc4(e3)   # (B, n, D/16, ...)

        # ---- Decoder with skip connections ----
        d3 = self.upsample(e4)                                  # → D/8
        d3 = self.dec3(torch.cat([d3, e3], dim=1))              # + skip e3

        d2 = self.upsample(d3)                                  # → D/4
        d2 = self.dec2(torch.cat([d2, e2], dim=1))              # + skip e2

        d1 = self.upsample(d2)                                  # → D/2
        d1 = self.dec1(torch.cat([d1, e1], dim=1))              # + skip e1

        # ---- SVF prediction at half resolution ----
        svf_half = self.pre_svf(d1)
        svf_half = self.flow_conv(svf_half)                     # (B, 3, D/2, ...)

        # ---- Integrate SVF → diffeomorphic phi (at half res) ----
        phi_half = self.integrator(svf_half)                    # (B, 3, D/2, ...)

        # ---- Upsample phi to full resolution + rescale magnitudes ----
        phi = self.upsample(phi_half) * 2.0                     # (B, 3, D, H, W)

        return phi

    def warp_image(self, m: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Warp intensity image m by displacement field phi (bilinear)."""
        return self._img_warper(m, phi)

    def warp_labels(self, s_m: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Warp label map s_m by displacement field phi (nearest-neighbor)."""
        return self._label_warper(s_m, phi)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
