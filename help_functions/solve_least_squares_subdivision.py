import torch
from help_functions.Sub_AT_torch import Sub_AT_torch
from help_functions.Sub_A_torch import Sub_A_torch

def apply_A_j(mask, x, j):
    for _ in range(j):
        x = Sub_A_torch(mask, x)
    return x

def apply_AT_j(mask, x, j):
    for _ in range(j):
        x = Sub_AT_torch(mask, x)
    return x

def apply_C_j(mask, x, j):
    # C*x = (AtA)x
    # (apply_C)*x = At(Ax)
    return apply_AT_j(mask, apply_A_j(mask, x, j), j)

def solve_least_squares_subdivision(
    z: torch.Tensor,
    mask: torch.Tensor,
    j = 1,
    max_iter=10000,
    tol=2
):
    """
    Решаем: min ||A d - z||^2
    """
    z = z.float()
    mask = mask.float()
    # dims = z.ndim
    # # начальная инициализация
    # slices = tuple(slice(None, None, 2) for _ in range(dims))
    # d = torch.zeros_like(z[slices])

    # r0 = A_j^T z
    r = z
    for _ in range(j):
        r = Sub_AT_torch(mask, r)

    d = torch.zeros_like(r)
    
    print(f'Shape of r {r.shape}')

    for k in range(max_iter):
        

        Cr = apply_C_j(mask, r, j)

        alpha = (r * r).sum() / (r * Cr).sum()

        d = d + alpha * r

        r_new = r - alpha * Cr
        
        print(f"ITER: {k}, grad norm = {torch.norm(r_new)}")

        if torch.norm(r_new) < tol:
            print(f"Converged at iter {k}")
            break

        r = r_new

    return d
