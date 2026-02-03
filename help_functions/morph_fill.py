from collections import deque
import torch

def morph_fill(I: torch.Tensor) -> torch.Tensor:
    
    I_clone = I.clone()
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
    
    I_clone[:] = 1 - I_clone
    return I_clone