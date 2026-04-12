import torch
import torch.nn.functional as F
from ..spatial import svf_generator, scaling_squaring, warper
from ... import config

def generate_shape_labels(batch_size=1, device=config.DEVICE):
    """
    Generates synthetic structural labels ("sm-shapes") via phase-1 IEEE method.
    
    Steps matched to Paper:
      1. We draw J smoothly varying noise images p_j by sampling voxels from
         a standard distribution at lower resolution r_p=1:32 and upsampling.
      2. Each image p_j is warped with a random deformation field phi_j.
         Each phi_j is obtained from SVF v_j ~ N(0, sigma_j^2) at lower res r_p,
         where sigma_j ~ U(0, b_p) with b_p=100.
      3. Assign label j corresponding to max intensity (Argmax).
    """
    J = config.SHAPES_NUM_LABELS
    target_shape = config.TARGET_SHAPE
    noise_res = config.SHAPES_NOISE_RES  # (5, 6, 7)
    
    # 1. Generate J noise volumes (standard normal)
    # shape: (B, J, D_r, H_r, W_r)
    noise_pj = torch.randn((batch_size, J, *noise_res), device=device)
    
    # Upsample p_j to full size
    noise_pj_up = F.interpolate(noise_pj, size=target_shape, mode='trilinear', align_corners=True)
    
    # 2. SVF Generation for each J independently
    # Vectorize by collapsing Batch and J into a single dimension B_prime
    B_prime = batch_size * J
    max_std = config.SHAPES_MAX_SVF_STD
    
    # Draw sigma_j ~ U(0, b_p) independently for each SVF
    sigma_j = torch.rand((B_prime, 1, 1, 1, 1), device=device) * max_std
    
    # Draw v_j ~ N(0, sigma_j^2) at lower resolution r_p
    v_j_low_res = torch.randn((B_prime, 3, *noise_res), device=device) * sigma_j
    
    # Upsample v_j to full size
    v_j = F.interpolate(v_j_low_res, size=target_shape, mode='trilinear', align_corners=True)
    
    # Integrate to get phi_j
    phi_j = scaling_squaring.integrate_svf(v_j, steps=config.SCALING_AND_SQUARING_STEPS)
    
    # Warp p_j using phi_j
    # Flatten noise to (B', 1, D, H, W)
    noise_pj_up_flat = noise_pj_up.view(B_prime, 1, *target_shape)
    warped_pj_flat = warper.warp_volume(noise_pj_up_flat, phi_j, mode='bilinear')
    
    # Reshape back to (B, J, D, H, W)
    warped_pj = warped_pj_flat.view(batch_size, J, *target_shape)
    
    # 3. Argmax for spatial partitions
    # Include a baseline 0 competitor for the background label.
    # Since warped_pj is standard normal, zeros acts as a natural mean threshold
    bg_prior = torch.zeros((batch_size, 1, *target_shape), device=device)
    
    combined_scores = torch.cat([bg_prior, warped_pj], dim=1) # (B, J+1, D, H, W)
    label_map = torch.argmax(combined_scores, dim=1, keepdim=True) # labels 0 to J
    
    return label_map.long()

