"""
Custom Shape-Focused Label Generator (Variant B)

Key Design — Streaming Argmax (Memory Efficient):
  Instead of stacking all J score channels as (B, J, D, H, W) on GPU (which
  would require ~17GB for J=26 at full resolution), we use an incremental
  "streaming argmax" strategy:
    - Compute phi (deformation field) once on GPU
    - Process each label j one at a time on CPU → transfer one (D,H,W) channel to GPU
    - Warp it → compare with running champion → update winner in place
    - GPU peak: phi + 3 single-channel tensors ≈ ~400MB total (vs 17GB naive)

Geometric Primitives (5 types, randomly chosen per label):
  - Sphere, Ellipsoid, Cuboid, Rotated Cuboid, Cylinder

Special Multi-Scale Deformations (stacked on top of 3-scale SVF):
  - Twist:   rotation swirl increasing with distance from center
  - Inflate: outward radial push from random center (tumor/ventricle)
  - Deflate: inward radial pull (complement of inflate)
  - Fold:    very-local high-std SVF patch (cortical crumpling)
"""

import math
import random
import torch
import torch.nn.functional as F
from ..spatial import svf_generator, scaling_squaring, warper
from ... import config


# -----------------------------------------------------------------------
# Coordinate grid helper
# -----------------------------------------------------------------------

def _make_coord_grid(target_shape, device):
    """
    Returns normalized [-1, 1] coordinate grids.
    Each output is (D, H, W).
    """
    D, H, W = target_shape
    zz = torch.linspace(-1, 1, D, device=device).view(D, 1, 1).expand(D, H, W)
    yy = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(D, H, W)
    xx = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(D, H, W)
    return zz, yy, xx


def _random_center():
    """Random center in [-0.7, 0.7] normalized coords."""
    return (
        random.uniform(-0.7, 0.7),
        random.uniform(-0.7, 0.7),
        random.uniform(-0.7, 0.7),
    )


def _random_size():
    return random.uniform(config.CUSTOM_PRIM_SIZE_MIN, config.CUSTOM_PRIM_SIZE_MAX)


# -----------------------------------------------------------------------
# Primitive rasterizers — run on GPU, output: (D, H, W) float32 in [0,1]
# -----------------------------------------------------------------------

def rasterize_sphere(target_shape, device):
    zz, yy, xx = _make_coord_grid(target_shape, device)
    cz, cy, cx = _random_center()
    r = _random_size()
    dist2 = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2
    return torch.exp(-dist2 / (2 * r**2))


def rasterize_ellipsoid(target_shape, device):
    zz, yy, xx = _make_coord_grid(target_shape, device)
    cz, cy, cx = _random_center()
    rz = _random_size(); ry = _random_size(); rx = _random_size()
    dist2 = ((zz - cz) / rz)**2 + ((yy - cy) / ry)**2 + ((xx - cx) / rx)**2
    return torch.exp(-dist2 / 2.0)


def rasterize_cuboid(target_shape, device):
    zz, yy, xx = _make_coord_grid(target_shape, device)
    cz, cy, cx = _random_center()
    hz = _random_size(); hy = _random_size(); hx = _random_size()
    k = 30.0  # sharpness
    return (torch.sigmoid(k * (hz - (zz - cz).abs()))
            * torch.sigmoid(k * (hy - (yy - cy).abs()))
            * torch.sigmoid(k * (hx - (xx - cx).abs())))


def rasterize_rotated_cuboid(target_shape, device):
    zz, yy, xx = _make_coord_grid(target_shape, device)
    cz, cy, cx = _random_center()
    angle = random.uniform(0, math.pi)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # Rotate in the xy plane
    xx_r = cos_a * (xx - cx) - sin_a * (yy - cy)
    yy_r = sin_a * (xx - cx) + cos_a * (yy - cy)
    hz = _random_size(); hy = _random_size(); hx = _random_size()
    k = 30.0
    return (torch.sigmoid(k * (hz - (zz - cz).abs()))
            * torch.sigmoid(k * (hy - yy_r.abs()))
            * torch.sigmoid(k * (hx - xx_r.abs())))


def rasterize_cylinder(target_shape, device):
    zz, yy, xx = _make_coord_grid(target_shape, device)
    cz, cy, cx = _random_center()
    r  = _random_size()
    hz = _random_size() * 1.5
    r_dist2 = (xx - cx)**2 + (yy - cy)**2
    s_r = torch.exp(-r_dist2 / (2 * r**2))
    s_z = torch.sigmoid(30.0 * (hz - (zz - cz).abs()))
    return s_r * s_z


_PRIMITIVE_FNS = [
    rasterize_sphere,
    rasterize_ellipsoid,
    rasterize_cuboid,
    rasterize_rotated_cuboid,
    rasterize_cylinder,
]


def _random_primitive(target_shape, device):
    """Pick a random primitive type and rasterize it on the target device."""
    return random.choice(_PRIMITIVE_FNS)(target_shape, device)


# -----------------------------------------------------------------------
# Blob helper (single channel, run on device)
# -----------------------------------------------------------------------

def _make_blob(target_shape, device):
    """Generate one blob channel on device. Returns (D, H, W) float32."""
    low = torch.rand((1, 1, *config.CUSTOM_BLOB_RES), dtype=torch.float32, device=device)
    up  = F.interpolate(low, size=target_shape, mode='trilinear', align_corners=True)
    # Repeated avg_pool approximates Gaussian
    k, p = 11, 5
    for _ in range(3):
        up = F.avg_pool3d(up, k, stride=1, padding=p)
    return up.squeeze(0).squeeze(0)   # (D, H, W)


# -----------------------------------------------------------------------
# Special deformation fields (on GPU)
# -----------------------------------------------------------------------

def _make_twist_svf(target_shape, device, batch_size):
    """
    Twist SVF: rotation swirl increasing with distance from z-axis.
    Simulates sulcal fold / gyrification patterns.
    """
    D, H, W = target_shape
    zz = torch.linspace(-1, 1, D, device=device).view(D, 1, 1).expand(D, H, W)
    yy = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(D, H, W)
    xx = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(D, H, W)

    r     = (xx**2 + yy**2).clamp(min=1e-6).sqrt()
    theta = random.uniform(0.3, config.CUSTOM_TWIST_STD) * r
    # Tangential velocity in xy-plane
    u_x = -torch.sin(theta) * (yy / r)
    u_y =  torch.cos(theta) * (xx / r)
    u_z =  torch.zeros_like(zz)

    svf = torch.stack([u_z, u_y, u_x], dim=0).unsqueeze(0)   # (1, 3, D, H, W)
    return svf.expand(batch_size, -1, -1, -1, -1).contiguous()


def _make_inflate_deflate_svf(target_shape, device, batch_size):
    """
    Radial inflate/deflate from a random local center.
    Models tumor growth, ventricular expansion, lung inflation.
    """
    D, H, W = target_shape
    zz = torch.linspace(-1, 1, D, device=device).view(D, 1, 1).expand(D, H, W)
    yy = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(D, H, W)
    xx = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(D, H, W)

    cz = random.uniform(-0.5, 0.5)
    cy = random.uniform(-0.5, 0.5)
    cx = random.uniform(-0.5, 0.5)
    dz = zz - cz; dy = yy - cy; dx = xx - cx
    r  = (dz**2 + dy**2 + dx**2).clamp(min=1e-6).sqrt()

    sign   = random.choice([-1.0, 1.0])   # inflate or deflate
    strength = sign * random.uniform(0.3, config.CUSTOM_INFLATE_STD)
    sigma  = random.uniform(0.2, 0.5)
    weight = torch.exp(-(r**2) / (2 * sigma**2))

    svf = torch.stack([
        strength * weight * (dz / r),
        strength * weight * (dy / r),
        strength * weight * (dx / r),
    ], dim=0).unsqueeze(0)               # (1, 3, D, H, W)
    return svf.expand(batch_size, -1, -1, -1, -1).contiguous()


def _make_fold_svf(target_shape, device, batch_size):
    """
    Local fold: very-low-res noise → sharp local crease after upsampling.
    Simulates cortical gyrification or cardiac muscle folding.
    """
    low = torch.randn((batch_size, 3, 3, 3, 3), device=device) * config.CUSTOM_FOLD_STD
    return F.interpolate(low, size=target_shape, mode='trilinear', align_corners=True)


# -----------------------------------------------------------------------
# Multi-scale SVF composition
# -----------------------------------------------------------------------

def _compose_multiscale_svf(target_shape, device, batch_size):
    """
    Compose SVFs at 3 spatial scales + one stochastic special deformation.
    Returns phi: (B, D, H, W, 3) grid-sample-ready deformation field.
    """
    # 1. Global: large-scale pose
    svf_g = svf_generator.generate_svf(
        target_shape, config.CUSTOM_SVF_GLOBAL_RES,
        config.CUSTOM_SVF_GLOBAL_STD, device, batch_size
    )
    # 2. Local: mid-scale
    svf_l = svf_generator.generate_svf(
        target_shape, config.CUSTOM_SVF_LOCAL_RES,
        config.CUSTOM_SVF_LOCAL_STD, device, batch_size
    )
    # 3. Micro: fine-scale inflate/deflate
    svf_m = svf_generator.generate_svf(
        target_shape, config.CUSTOM_SVF_MICRO_RES,
        config.CUSTOM_SVF_MICRO_STD, device, batch_size
    )

    # 4. Stochastic special deformation
    choice = random.choice(['twist', 'inflate', 'fold', 'none'])
    if choice == 'twist':
        svf_s = _make_twist_svf(target_shape, device, batch_size)
    elif choice == 'inflate':
        svf_s = _make_inflate_deflate_svf(target_shape, device, batch_size)
    elif choice == 'fold':
        svf_s = _make_fold_svf(target_shape, device, batch_size)
    else:
        svf_s = torch.zeros_like(svf_g)

    # Additive combination in velocity space, then integrate once
    svf_total = svf_g + svf_l + svf_m + svf_s
    phi = scaling_squaring.integrate_svf(svf_total, steps=config.SCALING_AND_SQUARING_STEPS)
    return phi


# -----------------------------------------------------------------------
# Main generator — Streaming Argmax
# -----------------------------------------------------------------------

def generate_custom_labels(batch_size=1, device=config.DEVICE):
    """
    Custom Shape-Focused Label Generator (Variant B).

    Memory-efficient streaming argmax strategy:
      - GPU peak: phi + 2 single-channel tensors (~400MB total)
      - CPU: one label channel at a time (never allocates J full volumes)

    Returns:
        torch.Tensor: (B, 1, D, H, W) long, labels in [0, J]
    """
    J            = config.CUSTOM_NUM_LABELS
    alpha        = config.CUSTOM_ALPHA
    beta         = config.CUSTOM_BETA
    target_shape = config.TARGET_SHAPE

    # --- Step 1: Compute deformation field once on GPU ---
    phi = _compose_multiscale_svf(target_shape, device, batch_size)

    # --- Step 2: Initialize running winner (background = label 0, score = 1.0) ---
    winner_score = torch.ones(
        (batch_size, 1, *target_shape), dtype=torch.float32, device=device
    )
    winner_idx = torch.zeros(
        (batch_size, 1, *target_shape), dtype=torch.long, device=device
    )

    # --- Step 3: Streaming argmax over J labels ---
    for j in range(J):
        # Rasterize primitive and blob dynamically on GPU (one channel)
        prim_gpu  = _random_primitive(target_shape, device)          # (D, H, W)
        blob_gpu  = _make_blob(target_shape, device)                 # (D, H, W)
        score_gpu = alpha * prim_gpu + beta * blob_gpu               # (D, H, W)

        # Add batch/channel dims
        score_gpu = score_gpu.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        if batch_size > 1:
            score_gpu = score_gpu.expand(batch_size, -1, -1, -1, -1).contiguous()

        # Warp with the shared phi
        score_warped = warper.warp_volume(score_gpu, phi, mode='bilinear')  # (B, 1, D, H, W)

        # Update champion via streaming argmax
        is_winner    = score_warped > winner_score                # (B, 1, D, H, W) bool
        winner_score = torch.where(is_winner, score_warped, winner_score)
        winner_idx   = torch.where(is_winner,
                                   torch.full_like(winner_idx, j + 1),
                                   winner_idx)

        # Explicit cleanup to keep VRAM strictly minimal
        del prim_gpu, blob_gpu, score_gpu, score_warped, is_winner

    del winner_score, phi
    return winner_idx   # (B, 1, D, H, W) long, labels in [0, J]
