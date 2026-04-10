import os
import torch
import numpy as np
import nibabel as nib
from src.models.network import SynthMorphUNet
from src.utils.metrics import dice_score
from src import config as cfg

def center_crop_numpy(vol, target_shape):
    d, h, w = vol.shape
    td, th, tw = target_shape
    start_d = (d - td) // 2
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return vol[start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = r"D:\NCKH\SynthMorph\runs\sm_shapes_20260403_105538"
    
    # We load the 2 models specified by the user
    checkpoints = ["checkpoint_best.pth", "checkpoint_0000020.pth"]
    
    # Set the fixed Atlas as subject 0447
    fixed_path = r"D:\NCKH\SynthMorph\data\raw\Eval\OASIS_OAS1_0447_MR1"
    f_img_np = nib.load(os.path.join(fixed_path, "aligned_norm.nii.gz")).get_fdata()
    f_seg_np = nib.load(os.path.join(fixed_path, "aligned_seg35.nii.gz")).get_fdata()
    
    # Crop to TARGET_SHAPE (160, 192, 224)
    target_shape = (160, 192, 224)
    f_img_np = center_crop_numpy(f_img_np, target_shape)
    f_seg_np = center_crop_numpy(f_seg_np, target_shape)
    
    # Normalize images
    f_img_np = np.clip(f_img_np, 0, 255) / 255.0
    
    f_img = torch.tensor(f_img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    f_seg = torch.tensor(f_seg_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Evaluate on the remaining 10 subjects (0448 to 0457)
    subjects = [f"04{i}" for i in range(48, 58)]
    
    for ckpt_name in checkpoints:
        print(f"\n{'='*50}")
        print(f"📊 EVALUATING MODEL: {ckpt_name}")
        print(f"{'='*50}")
        
        ckpt_path = os.path.join(run_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"⚠️ File not found: {ckpt_path}")
            continue
            
        # Initialize architecture identical to training run
        # Wait, the config inside the run was nb_features=256
        model = SynthMorphUNet(nb_features=256, integration_steps=5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
        model.eval()
        
        all_dices = []
        with torch.no_grad():
            for subj in subjects:
                moving_path = rf"D:\NCKH\SynthMorph\data\raw\Eval\OASIS_OAS1_{subj}_MR1"
                if not os.path.exists(moving_path):
                    continue
                    
                m_img_np = nib.load(os.path.join(moving_path, "aligned_norm.nii.gz")).get_fdata()
                m_seg_np = nib.load(os.path.join(moving_path, "aligned_seg35.nii.gz")).get_fdata()
                
                m_img_np = center_crop_numpy(m_img_np, target_shape)
                m_seg_np = center_crop_numpy(m_seg_np, target_shape)
                m_img_np = np.clip(m_img_np, 0, 255) / 255.0
                
                m_img = torch.tensor(m_img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                m_seg = torch.tensor(m_seg_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # Predict Registration Field phi
                phi = model(m_img, f_img)
                warped_m_seg = model.warp_labels(m_seg, phi)
                
                # Calculate Dice Score (Classes 0-35)
                # Note: Class 0 is background, so we exclude it later.
                dice_res = dice_score(warped_m_seg, f_seg, num_labels=36)
                
                # Foreground average (Labels 1 to 35)
                # Some labels might be empty, but dice.mean() includes all.
                fg_dice = dice_res['per_class'][1:].mean()
                
                print(f"Subject {subj} -> 0447: Mean Foreground Dice = {fg_dice:.4f}")
                all_dices.append(fg_dice)
                
        if all_dices:
            print(f"\n🚀 {ckpt_name} FINAL AVERAGE DICE: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}\n")

if __name__ == '__main__':
    main()
