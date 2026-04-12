import torch
import torch.nn.functional as F
from ... import config

def apply_bias_field(image, res=None, max_std=None):
    """
    Applies a synthetic low-frequency bias field to the given image to simulate MRI RF inhomogeneity.
    (I * e^B)
    """
    if res is None:
        res = config.BIAS_FIELD_RES
    if max_std is None:
        max_std = config.BIAS_FIELD_MAX_STD
        
    B, C, D, H, W = image.shape
    device = image.device
    out = image.clone()
    
    for b in range(B):
        # 1. Generate low res Gaussian noise. sigma_B ~ U(0, b_B)
        sigma_B = torch.empty(1).uniform_(0.0, max_std).item()
        low_res_noise = torch.randn((1, C, *res), device=device) * sigma_B
        
        # 2. Upsample back to image resolution smoothly
        bias_log = F.interpolate(
            low_res_noise, size=(D, H, W), mode='trilinear', align_corners=True
        )
        
        # 3. Exponentiate to create smooth multiplicative field
        bias_field = torch.exp(bias_log)
        
        # 4. Multiply with image
        out[b:b+1] = image[b:b+1] * bias_field
        
    return out
