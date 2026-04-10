import torch
import torch.nn.functional as F

def generate_svf(target_shape, svf_res, std, device, batch_size=1):
    """
    Generate a Stationary Velocity Field (SVF).
    - Samples Gaussian noise at a low resolution matching `svf_res`.
    - Upsamples to `target_shape`.
    - Multiplies by `std` to scale the displacement vector magnitudes.
    
    Args:
        target_shape (tuple): Spatial dimensions (H, W, D).
        svf_res (tuple): Low resolution shape (h, w, d) for sampling.
        std (float): Scaling factor for the velocities.
        device (torch.device): CPU or GPU.
        batch_size (int): Batch size.
        
    Returns:
        torch.Tensor: Vector field of shape (B, 3, H, W, D) representing displacements.
    """
    # Sample zero-mean, unit-variance Gaussian noise
    svf_low_res = torch.randn((batch_size, 3, *svf_res), device=device)
    
    # Scale intensities to specified standard deviation
    svf_low_res = svf_low_res * std
    
    # Upsample to full resolution.
    # Align corners = True is standard for displacement field upsampling to maintain fixed bounding planes.
    svf = F.interpolate(svf_low_res, size=target_shape, mode='trilinear', align_corners=True)
    
    return svf
