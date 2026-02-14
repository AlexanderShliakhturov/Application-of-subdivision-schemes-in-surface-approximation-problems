# from collections import deque
# import torch

# def morph_fill(I: torch.Tensor) -> torch.Tensor:
#     """    
#     2D: (H, W)
#     3D: (D, H, W)
#     """

#     I_clone = I.clone()
#     dims = I_clone.ndim

#     if dims == 2:
#         H, W = I_clone.shape
#         stack = deque([(0, 0)])

#         while stack:
#             x, y = stack.popleft()

#             if 0 <= x < H and 0 <= y < W:
#                 if I_clone[x, y] == 0:
#                     I_clone[x, y] = 1
#                     stack.append((x-1, y))
#                     stack.append((x+1, y))
#                     stack.append((x, y-1))
#                     stack.append((x, y+1))

#     elif dims == 3:
#         D, H, W = I_clone.shape
#         stack = deque([(0, 0, 0)])

#         while stack:
#             z, x, y = stack.popleft()

#             if 0 <= z < D and 0 <= x < H and 0 <= y < W:
#                 if I_clone[z, x, y] == 0:
#                     I_clone[z, x, y] = 1

#                     stack.append((z-1, x, y))
#                     stack.append((z+1, x, y))
#                     stack.append((z, x-1, y))
#                     stack.append((z, x+1, y))
#                     stack.append((z, x, y-1))
#                     stack.append((z, x, y+1))

#     else:
#         raise ValueError("Поддерживаются только 2D и 3D тензоры")

#     I_clone[:] = 1 - I_clone + I
#     return I_clone


from collections import deque
import torch
from tqdm import tqdm
from scipy import ndimage
import numpy as np


def morph_fill(I: torch.Tensor) -> torch.Tensor:
    """
    2D: (H, W)
    3D: (D, H, W)
    """

    if I.ndim not in (2, 3):
        raise ValueError("Поддерживаются только 2D и 3D тензоры")

    I_clone = I.clone()
    dims = I_clone.ndim

    total_pixels = I_clone.numel()

    start = (0,) * dims
    if I_clone[start] != 0:
        return I_clone

    stack = deque([start])
    I_clone[start] = 1  # помечаем сразу

    with tqdm(
        total=total_pixels,
        desc="Flood fill",
        unit="px",
    ) as pbar:

        pbar.update(1)

        if dims == 2:
            H, W = I_clone.shape

            while stack:
                x, y = stack.popleft()

                for nx, ny in (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ):
                    if 0 <= nx < H and 0 <= ny < W:
                        if I_clone[nx, ny] == 0:
                            I_clone[nx, ny] = 1
                            stack.append((nx, ny))
                            pbar.update(1)

        else:  # 3D
            D, H, W = I_clone.shape

            while stack:
                z, x, y = stack.popleft()

                for nz, nx, ny in (
                    (z - 1, x, y),
                    (z + 1, x, y),
                    (z, x - 1, y),
                    (z, x + 1, y),
                    (z, x, y - 1),
                    (z, x, y + 1),
                ):
                    if 0 <= nz < D and 0 <= nx < H and 0 <= ny < W:
                        if I_clone[nz, nx, ny] == 0:
                            I_clone[nz, nx, ny] = 1
                            stack.append((nz, nx, ny))
                            pbar.update(1)

    I_clone[:] = 1 - I_clone + I
    return I_clone



def morph_fill_fast(I: torch.Tensor) -> torch.Tensor:
    """
    Flood fill от (0,0) или (0,0,0)
    """

    I_np = I.cpu().numpy()

    mask = (I_np == 0)

    # Реконструкция от стартовой точки
    seed = np.zeros_like(mask)
    seed[(0,) * I_np.ndim] = mask[(0,) * I_np.ndim]

    filled = ndimage.binary_propagation(seed, mask=mask)

    result = 1- filled 
    return torch.from_numpy(result).to(I.device)



# def morph_fill_fast(I: torch.Tensor) -> torch.Tensor:
#     """
#     Заполнение внутренностей замкнутого контура.
#     Контур должен быть равен 1.
#     Работает для 2D и 3D.
#     """

#     device = I.device
#     dtype = I.dtype

#     I_np = I.detach().cpu().numpy().astype(bool)

#     filled = ndimage.binary_fill_holes(I_np)

#     # Сначала создаём torch-тензор
#     result = torch.from_numpy(filled)

#     # Затем приводим к исходному dtype
#     return result.to(device=device, dtype=dtype)