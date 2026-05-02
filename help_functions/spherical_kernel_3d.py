import torch

def spherical_kernel_3d(radius: int, dim = 3) -> torch.Tensor:
    size = 2 * radius + 1
    center = radius
    
    if dim == 3:
        z, y, x = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )
        dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2
        kernel = (dist_sq <= radius**2).float()
    elif dim == 2:
        y, x = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )
        dist_sq = (x - center)**2 + (y - center)**2
        kernel = (dist_sq <= radius**2).float()
        
    return kernel
