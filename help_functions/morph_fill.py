from collections import deque
import torch

def morph_fill(I: torch.Tensor) -> torch.Tensor:
    """    
    2D: (H, W)
    3D: (D, H, W)
    """

    I_clone = I.clone()
    dims = I_clone.ndim

    if dims == 2:
        H, W = I_clone.shape
        stack = deque([(0, 0)])

        while stack:
            x, y = stack.popleft()

            if 0 <= x < H and 0 <= y < W:
                if I_clone[x, y] == 0:
                    I_clone[x, y] = 1
                    stack.append((x-1, y))
                    stack.append((x+1, y))
                    stack.append((x, y-1))
                    stack.append((x, y+1))

    elif dims == 3:
        D, H, W = I_clone.shape
        stack = deque([(0, 0, 0)])

        while stack:
            z, x, y = stack.popleft()

            if 0 <= z < D and 0 <= x < H and 0 <= y < W:
                if I_clone[z, x, y] == 0:
                    I_clone[z, x, y] = 1

                    stack.append((z-1, x, y))
                    stack.append((z+1, x, y))
                    stack.append((z, x-1, y))
                    stack.append((z, x+1, y))
                    stack.append((z, x, y-1))
                    stack.append((z, x, y+1))

    else:
        raise ValueError("Поддерживаются только 2D и 3D тензоры")

    I_clone[:] = 1 - I_clone + I
    return I_clone
