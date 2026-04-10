import os
import tarfile
import requests
from tqdm import tqdm
from .. import config

def download_file(url, target_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(target_path, 'wb') as file, tqdm(
        desc=os.path.basename(target_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def extract_tar(tar_path, output_dir):
    with tarfile.open(tar_path, 'r:gz') as tar:
        print(f"Extracting {tar_path} into {output_dir}...")
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            # Skip symlinks and hardlinks which cause RecursionError in Windows tarfile
            if member.issym() or member.islnk():
                continue
            try:
                tar.extract(member, path=output_dir, set_attrs=False)
            except Exception as e:
                pass
        print("Extraction complete.")

def ensure_buckner_data():
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    tar_path = os.path.join(config.RAW_DATA_DIR, "tutorial_data.tar.gz")
    
    # Download
    if not os.path.exists(tar_path):
        print(f"Downloading FreeSurfer tutorial data from {config.BUCKNER40_URL}...")
        download_file(config.BUCKNER40_URL, tar_path)
    else:
        print("Tutorial data tarball already exists.")
        
    extract_dir = os.path.join(config.RAW_DATA_DIR, "tutorial_data")
    if not os.path.exists(extract_dir):
        extract_tar(tar_path, config.RAW_DATA_DIR)
    else:
        print("Tutorial data already extracted.")

    # Find aseg.mgz files recursively or known subject labels
    aseg_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('aseg.mgz') or file.endswith('aparc+aseg.mgz'):
                aseg_files.append(os.path.join(root, file))
    
    if not aseg_files:
        print("Warning: No aseg.mgz files found in the extracted tutorial data.")
    else:
        print(f"Found {len(aseg_files)} segmentation label files.")
    
    return aseg_files

if __name__ == "__main__":
    ensure_buckner_data()
