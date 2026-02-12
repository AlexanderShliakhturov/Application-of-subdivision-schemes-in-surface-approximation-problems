import torch
from help_functions.Sub_AT_torch import Sub_AT_torch
from help_functions.Sub_A_torch import Sub_A_torch


def apply_C(mask, x):
    
    
    # C*x = (AtA)x
    # (apply_C)*x = At(Ax)
    
    return Sub_AT_torch(mask, Sub_A_torch(mask, x))

def solve_least_squares_subdivision(
    z: torch.Tensor,
    mask: torch.Tensor,
    max_iter=1000,
    tol=1e-5
):
    """
    Решаем: min ||A d - z||^2
    """
    # dims = z.ndim
    # # начальная инициализация
    # slices = tuple(slice(None, None, 2) for _ in range(dims))
    # d = torch.zeros_like(z[slices])

    # r0 = A^T z
    r = Sub_AT_torch(mask, z)
    d = torch.zeros_like(r)
    
    print(f'Shape of r {r.shape}')

    for k in range(max_iter):
        
        Cr = apply_C(mask, r)
        alpha = (r * r).sum() / (r * Cr).sum()

        d = d + alpha * r

        r_new = r - alpha * Cr
        
        print(f"ITER: {k}, grad norm = {torch.norm(r_new)}")

        if torch.norm(r_new) < tol:
            print(f"Converged at iter {k}")
            break

        r = r_new

    return d
