import torch
import torch.nn.functional as F

def warp_volume(volume, deformation_field, mode='nearest'):
    """
    Warps a given 3D volume according to a deformation field using PyTorch's grid_sample.
    
    Args:
        volume (torch.Tensor): Tensor of shape (B, C, D, H, W).
        deformation_field (torch.Tensor): Coordinate grid of shape (B, D, H, W, 3).
        mode (str): Interpolation mode ('nearest', 'bilinear', 'bicubic').
                    'nearest' is strictly used for discrete label maps.
                    'bilinear' or 'bicubic' for intensity/spatial warpings.
                    
    Returns:
        torch.Tensor: Warped volume matching the shape of the input `volume`.
    """
    
    # Check dimensions
    if volume.ndim == 3: # (D, H, W) -> (1, 1, D, H, W)
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4: # (C, D, H, W) -> (1, C, D, H, W)
        volume = volume.unsqueeze(0)
        
    warped = F.grid_sample(
        volume.float(), # Needs to be float for grid_sample
        deformation_field,
        mode=mode,
        padding_mode='border',
        align_corners=True
    )
    
    if mode == 'nearest':
        warped = warped.round().long()
        
    return warped
