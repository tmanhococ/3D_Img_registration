import nibabel as nib
import torch
import numpy as np
from .. import config

def load_mgz(filepath, device=config.DEVICE):
    """
    Load a .mgz (or .nii.gz) file using nibabel and convert to a PyTorch tensor.
    Input must be integer label map.
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(data.astype(np.float32)).to(device)
    
    return tensor
