import torch
import numpy as np
import torch.nn.functional as F



def Sub_A_torch(mask: torch.Tensor, x_0: torch.Tensor):
    """
    x_0: torch.Tensor
       shape:
         1D: (L)
         2D: (H, W)
         3D: (D, H, W)

    mask: torch.Tensor
       shape:
         1D: (kd)
    """
    
    s = len(x_0.shape)
    if (s == 1):
        kernel = mask.clone()
        conv_func = F.conv1d
    elif (s == 2):
        kernel = torch.einsum('i,j->ij', mask, mask)
        conv_func = F.conv2d
    else:
        kernel = torch.einsum('i,j,k->ijk', mask, mask, mask)
        conv_func = F.conv3d
    
    #Переворот ядра
    kernel = kernel.flip(tuple(range(s)))
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    x = x_0.clone()
    Up_x = torch.zeros(1, 1, *(2 * np.array(x.shape)))

    #upsample + padd
    if (s==1):
        Up_x[..., ::2] = x[:]
    elif (s==2):
        Up_x[..., ::2, ::2] = x[:, :, ]
    else:
        Up_x[..., ::2, ::2, ::2] = x[:, :, :]
        
    # padding по каждой оси
    pads = []
    for k in reversed(kernel.shape[2:]):
        p = k // 2
        pads.extend([p, p])

    # pads в conv нужно добавлять в обратной последовательности, т.е W, H, D
    Up_x_padded = F.pad(Up_x, pads)
    
    conv_result = conv_func(Up_x_padded, kernel)
    
    return conv_result.squeeze(0).squeeze(0)[(slice(None, -1),) * (conv_result.ndim - 2)]
        