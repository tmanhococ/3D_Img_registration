import torch

def normalize_min_max(image):
    """
    Min-max normalization of an image to the [0, 1] range.
    Per-image normalization across the batch.
    
    Args:
        image: (B, C, D, H, W) tensor
    """
    B = image.shape[0]
    out = torch.zeros_like(image)
    
    for b in range(B):
        m = image[b].min()
        M = image[b].max()
        if M > m:
            out[b] = (image[b] - m) / (M - m)
        else:
            out[b] = image[b]
            
    return out

def apply_gamma(image, gamma_range):
    """
    Applies global gamma augmentation: I_new = I_old ^ gamma
    
    Args:
        image: (B, C, D, H, W) - Expects already normalized to [0, 1]
        gamma_range: tuple (min, max) to uniformly sample exponent.
    """
    B = image.shape[0]
    out = image.clone()
    
    for b in range(B):
        gamma = torch.empty(1).uniform_(*gamma_range).item()
        
        # Apply exponent explicitly handling tiny numbers near 0 smoothly.
        # Ensure we don't have negative numbers for floating point fractional powers.
        clamped = torch.clamp(out[b], min=0.0)
        out[b] = torch.pow(clamped, gamma)
        
    return out
