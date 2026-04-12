import os
import argparse
import csv
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# force UTF-8 output
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.models.network import SynthMorphUNet
from src.utils.metrics import dice_score

def center_crop_numpy(vol, target_shape):
    d, h, w = vol.shape
    td, th, tw = target_shape
    start_d = (d - td) // 2
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return vol[start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

def plot_grid(ax, phi_grid, slice_idx, spacing=6):
    """
    Vẽ Deformation Grid đúng chuẩn VoxelMorph (Wireframe đan chéo xanh đỏ).
    phi_grid: Tọa độ dịch chuyển tuyệt đối (displacement field) phi, shape: (3, D, H, W).
    """
    # Trích xuất lát cắt Coronal (Trục 1 - H). Phi có 3 kênh (W, H, D).
    # phi_disp = (3, D, H, W). X = W (index 2), Y = D (index 0). 
    # Khi cắt theo mặt H (slice_idx), chúng ta nhìn D và W.
    disp_X = phi_grid[2, :, slice_idx, :]  # shift theo chiều W (Width)
    disp_Y = phi_grid[0, :, slice_idx, :]  # shift theo chiều D (Depth)
    
    Y_dim, X_dim = disp_X.shape
    x = np.arange(0, X_dim, spacing)
    y = np.arange(0, Y_dim, spacing)
    X, Y = np.meshgrid(x, y)
    
    # Biến dạng tọa độ cục bộ X, Y. Phóng đại * -1 để khớp trục hệ tọa độ Matplotlib
    X_disp = X + disp_X[Y, X]
    Y_disp = Y + disp_Y[Y, X]
    
    # Vẽ các dải ngang (Màu xanh dương)
    for i in range(X_disp.shape[0]):
        ax.plot(X_disp[i,:], Y_disp[i,:], color='blue', linewidth=0.5, alpha=0.7)
    # Vẽ các dải dọc (Màu đỏ)
    for i in range(X_disp.shape[1]):
        ax.plot(X_disp[:,i], Y_disp[:,i], color='red', linewidth=0.5, alpha=0.7)
    
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SynthMorph on MRI")
    parser.add_argument("--model", type=str, default=r"runs/checkpoint_best_var_b.pth", help="Path to .pth checkpoint")
    parser.add_argument("--data-dir", type=str, default=r"data/raw/Eval", help="Dir with Eval MRI datasets")
    parser.add_argument("--out-dir", type=str, default=r"Result", help="Output directory")
    parser.add_argument("--fixed-subj", type=str, default="OASIS_OAS1_0447_MR1", help="Subject used as Atlas/Fixed")
    parser.add_argument("--moving-subj", type=str, default="OASIS_OAS1_0448_MR1", help="Subject used as Moving")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"\n" + "="*50)
    print(f"🧠 KHỞI CHẠY ĐÁNH GIÁ (1 CẶP MULTI-MODAL)")
    print(f"⚙️  Model: {args.model}")
    print(f"🎯  Fixed (aligned_orig): {args.fixed_subj}")
    print(f"🏃  Moving (aligned_norm): {args.moving_subj}")
    print(f"="*50 + "\n")
    
    model = SynthMorphUNet(nb_features=256, integration_steps=5).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()
    
    target_shape = (160, 192, 224)
    
    # 1. Tải Fixed Subject
    fixed_path = os.path.join(args.data_dir, args.fixed_subj)
    f_img_np = nib.load(os.path.join(fixed_path, "aligned_orig.nii.gz")).get_fdata()
    f_seg_np = nib.load(os.path.join(fixed_path, "aligned_seg35.nii.gz")).get_fdata()
    
    f_img_np = center_crop_numpy(f_img_np, target_shape)
    f_seg_np = center_crop_numpy(f_seg_np, target_shape)
    
    # BUG FIX 1: Dữ liệu Oasis MRI ở Evaluation Data đã nằm sẵn trong khoảng [0, 1]
    # Chia cho 255.0 tạo ra ảnh đen kịt -> Phi = 0 -> Dice không tăng không giảm
    if f_img_np.max() > 1.0:
        f_img_np = np.clip(f_img_np, 0, 255) / 255.0
    else:
        f_img_np = np.clip(f_img_np, 0, 1.0)
    
    f_img = torch.tensor(f_img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    f_seg = torch.tensor(f_seg_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 2. Tải Moving Subject
    moving_path = os.path.join(args.data_dir, args.moving_subj)
    m_img_np = nib.load(os.path.join(moving_path, "aligned_norm.nii.gz")).get_fdata()
    m_seg_np = nib.load(os.path.join(moving_path, "aligned_seg35.nii.gz")).get_fdata()
    
    m_img_np = center_crop_numpy(m_img_np, target_shape)
    m_seg_np = center_crop_numpy(m_seg_np, target_shape)
    
    if m_img_np.max() > 1.0:
        m_img_np = np.clip(m_img_np, 0, 255) / 255.0
    else:
        m_img_np = np.clip(m_img_np, 0, 1.0)
    
    m_img = torch.tensor(m_img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    m_seg = torch.tensor(m_seg_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    print("🚀 Đang tính toán đăng ký toàn bộ 3D Volume...")
    with torch.no_grad():
        phi = model(m_img, f_img)
        warped_m_seg = model._label_warper(m_seg, phi)
        warped_m_img = model.warp_image(m_img, phi)
        
        base_dice_res = dice_score(m_seg, f_seg, num_labels=36)
        reg_dice_res  = dice_score(warped_m_seg, f_seg, num_labels=36)
        
        base_dice = base_dice_res['per_class'][1:].mean().item()
        reg_dice  = reg_dice_res['per_class'][1:].mean().item()
        improvement = reg_dice - base_dice
        
    print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ (TOÀN BỘ THỂ TÍCH 3D):")
    print(f"   => Dice Ban đầu (Baseline):    {base_dice:.4f}")
    print(f"   => Dice Sau đăng ký (Warped):  {reg_dice:.4f}")
    print(f"   => Mức cải thiện:              {improvement:+.4f}\n")
    
    # -------------------------------------------------------------
    # TRỰC QUAN HÓA XUẤT RA ẢNH DẠNG GRAYSCALE CHUẨN VOXELMORPH
    # -------------------------------------------------------------
    print("📸 Đang trích xuất mặt cắt đồ họa chuẩn VoxelMorph (Coronal Plane)...")
    
    # FIX 2: Cắt não theo mặt phẳng Y (Coronal) giống hệt Voxelmorph để coi hai thùy não
    slice_idx = target_shape[1] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='black')
    
    # 1. Fixed
    axes[0].imshow(f_img_np[:, slice_idx, :].T, cmap='gray', origin='lower')
    axes[0].set_title(f"1. Fixed Image (Atlas: {args.fixed_subj})", color='white', pad=10, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Moving
    axes[1].imshow(m_img_np[:, slice_idx, :].T, cmap='gray', origin='lower')
    axes[1].set_title(f"2. Moving Image (Patient: {args.moving_subj})\nBase Dice: {base_dice:.3f}", color='white', pad=10, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Warped
    w_img_slice = warped_m_img[0, 0, :, slice_idx, :].cpu().numpy()
    axes[2].imshow(w_img_slice.T, cmap='gray', origin='lower')
    axes[2].set_title(f"3. Warped Image\nReg Dice: {reg_dice:.3f} ({improvement:+.3f})", color='white', pad=10, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Deformation Grid
    phi_disp = phi[0].cpu().numpy()  # (3, D, H, W)
    plot_grid(axes[3], phi_disp, slice_idx, spacing=5)
    axes[3].set_title("4. Deformation Grid", color='white', pad=10, fontweight='bold')
    
    out_path = os.path.join(args.out_dir, f"result_crossmodal_0447_0448.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='black')
    plt.close(fig)
    print(f"💾 Đã lưu ảnh kết quả sạch sẽ tại: {out_path}")

if __name__ == '__main__':
    main()

