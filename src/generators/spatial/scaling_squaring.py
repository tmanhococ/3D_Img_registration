import torch
import torch.nn.functional as F

def create_identity_grid(target_shape, device):
    """
    Creates an absolute unnormalized coordinate grid [0, H-1] x [0, W-1] x [0, D-1].
    Or normalized [-1, 1]. F.grid_sample requires normalized [-1, 1].
    
    Returns grid of shape (1, H, W, D, 3) where the last dimension holds coordinates (x, y, z).
    Note: PyTorch grid_sample expects coordinates in (x, y, z) order (W, H, D or D, W, H depending).
    In PyTorch 3D: (x, y, z) -> spatial dimensions: depth, height, width.
    Actually, PyTorch 5D grid_sample grid expects:
    grid[..., 0] = x (width, dimension -1)
    grid[..., 1] = y (height, dimension -2)
    grid[..., 2] = z (depth, dimension -3)
    """
    d, h, w = target_shape[0], target_shape[1], target_shape[2]
    # Keep strictly -1 to 1 based on dimensions
    vectors = [
        torch.linspace(-1, 1, steps=d, device=device), # depth
        torch.linspace(-1, 1, steps=h, device=device), # height
        torch.linspace(-1, 1, steps=w, device=device)  # width
    ]
    # Meshgrid ordering: 'ij' keeps grid shape (d, h, w)
    grid_d, grid_h, grid_w = torch.meshgrid(vectors, indexing='ij')
    
    # Stack along last dim, PyTorch wants (W, H, D) -> (grid_w, grid_h, grid_d) for x, y, z
    grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)
    return grid.unsqueeze(0) # (1, d, h, w, 3)


def integrate_svf(svf, steps=7):
    """
    Integrates an SVF using Scaling and Squaring to produce a diffeomorphic deformation field.
    
    Args:
        svf (torch.Tensor): SVF tensor of shape (B, 3, D, H, W).
        steps (int): Number of scaling and squaring steps.
        
    Returns:
        torch.Tensor: Deformation field grid of shape (B, D, H, W, 3) ready for F.grid_sample.
    """
    b, _, d, h, w = svf.shape
    device = svf.device
    target_shape = (d, h, w)
    
    # 1. Scale
    scaled_svf = svf / (2 ** steps)
    
    # 2. To F.grid_sample compatible format: shape (B, D, H, W, 3).
    # scaled_svf comes as (B, 3, D, H, W). We must permute to (B, D, H, W, 3)
    # AND reverse the 3 channels to represent (width, height, depth) shifts.
    shift = scaled_svf.permute(0, 2, 3, 4, 1) # (B, D, H, W, 3)
    
    # Actually, svf models spatial displacements. Since spatial domain is [-1, 1],
    # the vector field needs scaling relative to dimension sizes if it's in absolute pixels,
    # but let's assume `scaled_svf` directly represents shifts in the [-1, 1] normalized space.
    # Note: If `svf` is originally bounded or interpreted as pixel offsets, then:
    # shift_normalized[..., 0] = shift[..., 2] / (W - 1) * 2
    # shift_normalized[..., 1] = shift[..., 1] / (H - 1) * 2
    # shift_normalized[..., 2] = shift[..., 0] / (D - 1) * 2
    # Let's perform uniform normalized space shifting: Assume svf is already generated in appropriate scaled variance.
    # The standard way to represent vector fields: channel 0 for 'D', 1 for 'H', 2 for 'W'
    shift_grid = shift.clone()
    shift_grid[..., 0] = shift[..., 2] / (w/2)  # x shift
    shift_grid[..., 1] = shift[..., 1] / (h/2)  # y shift
    shift_grid[..., 2] = shift[..., 0] / (d/2)  # z shift
    
    # Base identity
    identity_grid = create_identity_grid(target_shape, device)
    
    # Initial deformation
    phi = identity_grid.expand(b, -1, -1, -1, -1) + shift_grid
    
    # 3. Square (compose)
    for _ in range(steps):
        # Composition: phi = phi(phi)
        # However, F.grid_sample wraps 'phi' sampled at locations of 'phi'.
        # F.grid_sample expects input shape (N, C, D, H, W) and grid (N, D, H, W, 3)
        # phi is (N, D, H, W, 3), we need to sample phi itself!
        # Convert phi to PyTorch channel-first format to be sampled
        phi_as_image = phi.permute(0, 4, 1, 2, 3) # (N, 3, D, H, W)
        phi_sampled = F.grid_sample(phi_as_image, phi, mode='bilinear', padding_mode='border', align_corners=True)
        # Re-convert back to grid layout
        phi = phi_sampled.permute(0, 2, 3, 4, 1) # (N, D, H, W, 3)
    
    return phi
