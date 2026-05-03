import torch
import math
import torch.nn.functional as F

def nearest_multiple_of_n(x, n):
    return n * math.ceil(x / n)

nearest_multiple_of_n(713, 8)

def pad_to_multiple_centered(x: torch.Tensor, j: int):
    m = 2 ** j
    d0, d1, d2 = x.shape

    def compute_pad(d):
        new_d = ((d + m - 1) // m) * m
        p = new_d - d
        left = p // 2
        right = p - left
        return left, right

    p0l, p0r = compute_pad(d0)
    p1l, p1r = compute_pad(d1)
    p2l, p2r = compute_pad(d2)

    pads = (p2l, p2r, p1l, p1r, p0l, p0r)

    x_padded = F.pad(x, pads)

    return x_padded, (p0l, p0r, p1l, p1r, p2l, p2r)