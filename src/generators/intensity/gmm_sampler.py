import torch

def sample_intensities(label_map, mean_range, std_range):
    """
    Simulates MRI intensities by sampling Mean and Variance from a uniform distribution 
    for each present discrete structural label, assigning a random Gaussian mask.
    
    Args:
        label_map: (B, 1, D, H, W) IntTensor
        mean_range: tuple (min, max) for uniform distribution of means
        std_range: tuple (min, max) for uniform distribution of variances/std devs
        
    Returns:
        torch.Tensor: Synthetic intensity volume
    """
    B, C, D, H, W = label_map.shape
    device = label_map.device
    out_img = torch.zeros_like(label_map, dtype=torch.float32)
    
    for b in range(B):
        # find unique labels in this batch item
        curr_map = label_map[b:b+1]
        unique_labels = torch.unique(curr_map)
        
        for label in unique_labels:
            if label == 0:
                # Keep background pure 0
                continue
                
            mu_j = torch.empty(1).uniform_(*mean_range).item()
            sigma_j = torch.empty(1).uniform_(*std_range).item()
            
            # Mask for current label
            mask = (curr_map == label).float()
            
            # Sample independent noise for all voxels, then multiply by mask
            noise = torch.randn(curr_map.shape, device=device) * sigma_j + mu_j
            out_img[b:b+1] += mask * noise
            
    return out_img
