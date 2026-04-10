import torch
import torch.nn.functional as F
from ... import config

def apply_bias_field(image, res=config.BIAS_FIELD_RES, std=config.BIAS_FIELD_STD):
    """
    Applies a synthetic low-frequency bias field to the given image to simulate MRI RF inhomogeneity.
    (I * e^B)
    
    Args:
        image: (B, C, D, H, W)
        res: Tuple defining the resolution of the noise grid
        std: Standard deviation of multiplicative bias field
        
    Returns:
        torch.Tensor: Bias-corrupted image
    """
    B, C, D, H, W = image.shape
    device = image.device
    
    # 1. Generate low res Gaussian noise
    low_res_noise = torch.randn((B, C, *res), device=device) * std
    
    # 2. Upsample back to image resolution smoothly
    bias_log = F.interpolate(
        low_res_noise, size=(D, H, W), mode='trilinear', align_corners=True
    )
    
    # 3. Exponentiate to create smooth multiplicative field
    bias_field = torch.exp(bias_log)
    
    # 4. Multiply with image
    return image * bias_field
