import torch
import torch.nn.functional as F

B = 1
J = 26
device = 'cuda' if torch.cuda.is_available() else 'cpu'
res = (5, 6, 7)
target_shape = (160, 192, 224)

max_vals = torch.full((B, 1, *target_shape), -float('inf'), device=device)

print(f"Testing with device {device} over {J} labels...")
for _ in range(J):
    n = torch.randn((B, 1, *res), device=device)
    u = F.interpolate(n, size=target_shape, mode='trilinear', align_corners=True)
    max_vals = torch.max(max_vals, u)

for mu0 in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
    bg_ratio = (max_vals < mu0).float().mean().item()
    print(f"mu0={mu0:.1f} -> Background {bg_ratio*100:.2f}%")
