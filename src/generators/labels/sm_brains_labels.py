import os
import random
import glob
import torch
from ..data import loader, preprocess, buckner_downloader
from ... import config

def get_random_aseg(device=config.DEVICE):
    """
    Randomly selects, loads, and preprocesses an anatomical label map from the downloaded Buckner40 dataset.
    
    Returns:
        torch.Tensor: Preprocessed label map of shape (1, 1, D, H, W) containing integer labels.
    """
    extract_dir = os.path.join(config.RAW_DATA_DIR, "tutorial_data")
    
    # Check if data exists, if not download
    if not os.path.exists(extract_dir):
        buckner_downloader.ensure_buckner_data()
        
    # Search all possible segments
    aseg_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('aseg.mgz') or file.endswith('aparc+aseg.mgz'):
                aseg_files.append(os.path.join(root, file))
                
    if not aseg_files:
        raise FileNotFoundError("Buckner40/FreeSurfer 'aseg.mgz' files could not be found.")

    # Select random file
    chosen_file = random.choice(aseg_files)
    
    # Load raw integer tensor
    label_map = loader.load_mgz(chosen_file, device=device)
    
    # Preprocess (Resample to TARGET_SHAPE using nearest-neighbor)
    preprocessed_map = preprocess.resample_label_map(label_map, config.TARGET_SHAPE)
    
    # Expand dims for batch and channel
    preprocessed_map = preprocessed_map.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
    
    return preprocessed_map
