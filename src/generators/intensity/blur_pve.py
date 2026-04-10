import torch
import torch.nn.functional as F
import math

def gaussian_kernel_1d(sigma, kernel_size):
    """Generates a 1D Gaussian kernel."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    kernel = torch.exp(- (x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def apply_anisotropic_blur(image, std_range):
    """
    Simulates Partial Volume Effects (PVE) by applying anisotropic Gaussian blur.
    
    Args:
        image: (B, C, D, H, W)
        std_range: tuple limiting the sigma for independent x, y, z blurs.
    """
    B, C, D, H, W = image.shape
    device = image.device
    out = image.clone()
    
    for b in range(B):
        for c in range(C):
            # Sample 3 independent sigmas for D, H, W (z, y, x)
            sigmas = [torch.empty(1).uniform_(*std_range).item() for _ in range(3)]
            
            for dim, sigma in enumerate(sigmas):
                kernel_size = int(math.ceil(sigma * 3)) * 2 + 1
                if kernel_size < 3:
                    continue
                
                k = gaussian_kernel_1d(sigma, kernel_size).to(device)
                
                # Reshape for appropriate 3D convolution axis
                if dim == 0:   # Depth (D)
                    k = k.view(1, 1, -1, 1, 1)
                elif dim == 1: # Height (H)
                    k = k.view(1, 1, 1, -1, 1)
                else:          # Width (W)
                    k = k.view(1, 1, 1, 1, -1)
                    
                # Apply 1D conv iteratively
                img_slice = out[b:b+1, c:c+1]
                # 'same' padding requires odd kernel size, which we guaranteed
                padding = kernel_size // 2
                
                if dim == 0:
                    out[b:b+1, c:c+1] = F.conv3d(img_slice, k, padding=(padding, 0, 0))
                elif dim == 1:
                    out[b:b+1, c:c+1] = F.conv3d(img_slice, k, padding=(0, padding, 0))
                else:
                    out[b:b+1, c:c+1] = F.conv3d(img_slice, k, padding=(0, 0, padding))
            
    return out
