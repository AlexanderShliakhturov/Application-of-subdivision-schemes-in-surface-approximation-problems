import torch
import torch.nn.functional as F

def make_conv_2d_torch(source: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    
    source = source.float()
    kernel = kernel.float()
    
    """
    source: torch.Tensor, shape (H, W)
    kernel: torch.Tensor, shape (kH, kW)

    return:
        result: torch.Tensor, shape (H, W)
    """

    # риведение к формату conv2d
    # source: (1, 1, H, W)
    x = source.unsqueeze(0).unsqueeze(0)

    # kernel: (1, 1, kH, kW)
    w = kernel.unsqueeze(0).unsqueeze(0)

    #padding
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h))

    y = F.conv2d(x_padded, w)

    y = torch.clamp(y, max=1.0)

    result = y.squeeze(0).squeeze(0)

    return result
