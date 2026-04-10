import torch
import torch.nn.functional as F

def resample_label_map(label_tensor, target_shape):
    """
    Resamples a 3D label map to a target shape using nearest-neighbor interpolation.
    
    Args:
        label_tensor (torch.Tensor): 3D tensor of shape (H, W, D) or (1, 1, H, W, D).
        target_shape (tuple): Target shape like (160, 192, 224).
        
    Returns:
        torch.Tensor: Resampled 3D label map of shape (H_out, W_out, D_out).
    """
    if label_tensor.ndim == 3:
        # Add batch and channel dims: (1, 1, H, W, D)
        label_tensor = label_tensor.unsqueeze(0).unsqueeze(0)
    elif label_tensor.ndim == 4:
        # Add batch dim
        label_tensor = label_tensor.unsqueeze(0)

    # Note: Use nearest downsampling/upsampling to preserve discrete integer labels
    # Using float tensor for interpolation
    is_integer = False
    if not label_tensor.is_floating_point():
        label_tensor = label_tensor.float()
        is_integer = True

    resampled = F.interpolate(label_tensor, size=target_shape, mode='nearest')
    
    if is_integer:
        resampled = resampled.round().long()
    
    # Remove batch and channel dims
    return resampled.squeeze(0).squeeze(0)
