import torch
import numpy as np
import torch.nn.functional as F

def Sub_AT_torch(mask: torch.Tensor, y: torch.Tensor):
    
    # A*x = (subdivision)(upsample)x
    # At*x = (downsample)(subdivision_t)
    
    """
    y: torch.Tensor
       shape:
         1D: (L)
         2D: (H, W)
         3D: (D, H, W)

    mask: torch.Tensor
       shape:
         1D: (kd)
    """

    s = len(y.shape)

    # ядро a^T
    if s == 1:
        kernel = mask.clone
        conv_func = F.conv1d
    if s == 2:
        kernel = torch.einsum('i,j->ij', mask, mask)
        conv_func = F.conv2d
    else:
        kernel = torch.einsum('i,j,k->ijk', mask, mask, mask)
        conv_func = F.conv3d

    #Ядро уже не переворачиваем
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    y = y.unsqueeze(0).unsqueeze(0)

    # padding
    pads = []
    for k in reversed(kernel.shape[2:]):
        p = k // 2
        pads.extend([p, p])

    y_padded = F.pad(y, pads)

    conv = conv_func(y_padded, kernel)

    conv = conv.squeeze(0).squeeze(0)

    # downsampling
    if s == 1:
        return conv[::2]
    elif s == 2:
        return conv[::2, ::2]
    else: 
        return conv[::2, ::2, ::2]