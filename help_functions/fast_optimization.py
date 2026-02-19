import torch
import torch.nn.functional as F


def build_kernel(mask: torch.Tensor, dim: int, device=None):
    mask = mask.to(device)

    if dim == 1:
        kernel = mask
    elif dim == 2:
        kernel = torch.outer(mask, mask)
    elif dim == 3:
        kernel = torch.einsum('i,j,k->ijk', mask, mask, mask)
    else:
        raise ValueError("Only 1D, 2D, 3D supported")

    # conv_transpose не требует flip
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel


# ============================================================
#  A operator  (upsample + filter)
#  реализован через conv_transpose
# ============================================================

def Sub_A_fast(x, kernel):
    dim = x.ndim
    padding = kernel.shape[-1] // 2

    x = x.unsqueeze(0).unsqueeze(0)

    if dim == 1:
        out = F.conv_transpose1d(x, kernel, stride=2, padding=padding)
    elif dim == 2:
        out = F.conv_transpose2d(x, kernel, stride=2, padding=padding)
    else:
        out = F.conv_transpose3d(x, kernel, stride=2, padding=padding)

    return out.squeeze(0).squeeze(0)


# ============================================================
#  A^T operator (filter + downsample)
# ============================================================

def Sub_AT_fast(x, kernel):
    dim = x.ndim
    padding = kernel.shape[-1] // 2

    x = x.unsqueeze(0).unsqueeze(0)

    if dim == 1:
        out = F.conv1d(x, kernel, stride=2, padding=padding)
    elif dim == 2:
        out = F.conv2d(x, kernel, stride=2, padding=padding)
    else:
        out = F.conv3d(x, kernel, stride=2, padding=padding)

    return out.squeeze(0).squeeze(0)


# ============================================================
#  A^j и (A^T)^j
# ============================================================

def apply_A_j(x, kernel, j):
    for _ in range(j):
        x = Sub_A_fast(x, kernel)
    return x


def apply_AT_j(x, kernel, j):
    for _ in range(j):
        x = Sub_AT_fast(x, kernel)
    return x


def apply_C_j(x, kernel, j):
    # C = (A^T)^j A^j
    return apply_AT_j(apply_A_j(x, kernel, j), kernel, j)


# ============================================================
#  Solver (Steepest Descent)
# ============================================================

def solve_least_squares_subdivision_fast(
    z: torch.Tensor,
    mask: torch.Tensor,
    j=1,
    max_iter=500,
    tol=1e-6,
    device="cpu"
):
    device = torch.device(device)

    z = z.float().to(device)
    mask = mask.float().to(device)

    dim = z.ndim
    kernel = build_kernel(mask, dim, device)

    # r0 = (A^T)^j z
    r = apply_AT_j(z, kernel, j)

    d = torch.zeros_like(r)

    for k in range(max_iter):

        Cr = apply_C_j(r, kernel, j)

        denom = (r * Cr).sum()
        if torch.abs(denom) < 1e-20:
            break

        alpha = (r * r).sum() / denom

        d = d + alpha * r
        r_new = r - alpha * Cr

        grad_norm = torch.norm(r_new)

        print(f"ITER {k}: grad norm = {grad_norm:.6e}")

        if grad_norm < tol:
            print(f"Converged at iter {k}")
            break

        r = r_new

    return d

def solve_least_squares_subdivision_CG(
    z: torch.Tensor,
    mask: torch.Tensor,
    j=1,
    max_iter=200,
    tol=1e-6,
    device="cpu"
):
    device = torch.device(device)

    z = z.float().to(device)
    mask = mask.float().to(device)

    dim = z.ndim
    kernel = build_kernel(mask, dim, device)

    # b = (A^T)^j z
    b = apply_AT_j(z, kernel, j) 

    # начальное приближение
    x = torch.zeros_like(b)

    # r0 = b - Cx0
    r = b.clone()  # потому что x0 = 0
    p = r.clone()

    rs_old = torch.sum(r * r)

    for k in range(max_iter):

        Cp = apply_C_j(p, kernel, j)

        denom = torch.sum(p * Cp)
        if torch.abs(denom) < 1e-20:
            break

        alpha = rs_old / denom

        x = x + alpha * p
        r = r - alpha * Cp

        rs_new = torch.sum(r * r)

        grad_norm = torch.sqrt(rs_new)
        print(f"CG ITER {k}: residual = {grad_norm:.6e}")

        if grad_norm < tol:
            print(f"CG Converged at iter {k}")
            break

        beta = rs_new / rs_old
        p = r + beta * p

        rs_old = rs_new

    return x