import torch
import torch.nn.functional as F
from ..spatial import svf_generator, scaling_squaring, warper
from ... import config

def generate_shape_labels(batch_size=1, device=config.DEVICE):
    """
    Generates synthetic structural labels ("sm-shapes") via random blob noise and spatial deformations.
    
    Steps:
      1. Generates `config.SHAPES_NUM_LABELS` independent noise volumes at `config.SHAPES_NOISE_RES`
      2. Upsamples noise to `config.TARGET_SHAPE`
      3. Smooths with uniform/gaussian blur (F.avg_pool3d applied sequentially)
      4. Samples random SVF at `config.SHAPES_SVF_RES`
      5. Integrates SVF to single spatial Deformation field
      6. Warps upsampled noise fields
      7. Applies ArgMax to form discrete disjoint spatial clusters.
      
    Returns:
        torch.Tensor: Synthetic Label mask of shape (B, 1, D, H, W) containing integer labels 0 to J.
    """
    J = config.SHAPES_NUM_LABELS
    target_shape = config.TARGET_SHAPE
    
    # 1. Generate J noise volumes
    noise_res = config.SHAPES_NOISE_RES
    # (B, J, d, h, w)
    noise = torch.rand((batch_size, J, *noise_res), device=device)
    
    # 2. Upsample and Smooth
    # Note: Upsampling uniform noise natively introduces some smoothness (bilinear).
    noise_up = F.interpolate(noise, size=target_shape, mode='trilinear', align_corners=True)
    
    # 3. Add heavy smoothing to make blobby structures
    kernel_size = 11
    padding = kernel_size // 2
    for _ in range(3): # Repeated uniform smoothing approximates Gaussian
        noise_up = F.avg_pool3d(noise_up, kernel_size, stride=1, padding=padding)
        
    # 4. Generate random SVF for warping shapes
    # Create svf (B, 3, d, h, w) then integrate
    svf = svf_generator.generate_svf(
        target_shape=target_shape,
        svf_res=config.SHAPES_SVF_RES,
        std=config.SHAPES_SVF_STD,
        device=device,
        batch_size=batch_size
    )
    
    # 5. Integrate SVF -> Deformation Field
    deformation = scaling_squaring.integrate_svf(svf, steps=config.SCALING_AND_SQUARING_STEPS)
    
    # 6. Warp smoothed noise volumes
    # Use 'bilinear' because input noise is continuous probabilities. We argmax after.
    warped_noise = warper.warp_volume(noise_up, deformation, mode='bilinear')
    
    # 7. Argmax for spatial partitions
    # Include a flat background prior to Argmax (e.g., constant 0.5 likelihood everywhere for background)
    # Allows a "background" label 0 to dominate where no shapes peak.
    bg_prior = torch.ones((batch_size, 1, *target_shape), device=device) * 0.5 
    
    combined_scores = torch.cat([bg_prior, warped_noise], dim=1) # (B, J+1, D, H, W)
    label_map = torch.argmax(combined_scores, dim=1, keepdim=True) # (B, 1, D, H, W)
    
    return label_map.long()
