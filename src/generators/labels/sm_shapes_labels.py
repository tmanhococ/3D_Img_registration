import torch
import torch.nn.functional as F
from ..spatial import svf_generator, scaling_squaring, warper
from ... import config

def generate_shape_labels(batch_size=1, device=config.DEVICE):
    """
    Generates synthetic structural labels ("sm-shapes") via phase-1 IEEE method.
    Uses memory-efficient streaming argmax strategy to prevent OOM when J=26.
    """
    J = config.SHAPES_NUM_LABELS
    target_shape = config.TARGET_SHAPE
    noise_res = config.SHAPES_NOISE_RES  # (5, 6, 7)
    max_std = config.SHAPES_MAX_SVF_STD
    
    # Initialize running winner. 
    # PAPER SECRET: "we implicitly define the background state p0 to have constant intensity mu_0 > 0"
    # Since p_j ~ N(0, 1), a threshold of 1.5 to 2.0 ensures distinct geometric blobs instead of a dense soup.
    mu_0 = 2.0  
    winner_score = torch.ones(
        (batch_size, 1, *target_shape), dtype=torch.float32, device=device
    ) * mu_0
    winner_idx = torch.zeros(
        (batch_size, 1, *target_shape), dtype=torch.long, device=device
    )
    
    # Loop over J labels to maintain small ~400MB memory footprint instead of ~8GB
    for j in range(J):
        # 1. Generate 1 noise volume
        noise_pj = torch.randn((batch_size, 1, *noise_res), device=device)
        noise_pj_up = F.interpolate(noise_pj, size=target_shape, mode='trilinear', align_corners=True)
        
        # 2. SVF Generation for current label j
        sigma_j = torch.rand((batch_size, 1, 1, 1, 1), device=device) * max_std
        v_j_low = torch.randn((batch_size, 3, *noise_res), device=device) * sigma_j
        
        # Upsample SVF & integrate
        v_j = F.interpolate(v_j_low, size=target_shape, mode='trilinear', align_corners=True)
        phi_j = scaling_squaring.integrate_svf(v_j, steps=config.SCALING_AND_SQUARING_STEPS)
        
        # Warp noise pj
        warped_pj = warper.warp_volume(noise_pj_up, phi_j, mode='bilinear')
        
        # 3. Streaming Argmax update
        is_winner = warped_pj > winner_score
        winner_score = torch.where(is_winner, warped_pj, winner_score)
        winner_idx = torch.where(is_winner, 
                                 torch.full_like(winner_idx, j + 1), 
                                 winner_idx)
                                 
        # Memory cleanup
        del noise_pj, noise_pj_up, v_j_low, v_j, phi_j, warped_pj, is_winner
        
    del winner_score
    return winner_idx

