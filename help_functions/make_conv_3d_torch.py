import torch
import torch.nn.functional as F

def make_conv_3d_torch(source: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    source: torch.Tensor, shape (D, H, W)
    kernel: torch.Tensor, shape (kD, kH, kW)

    return:
        result: torch.Tensor, shape (D, H, W)
    """

    source = source.float()
    kernel = kernel.float()
    
    device = source.device
    
    kernel = kernel.to(device)

    # conv3d формат
    # source: (N, C, D, H, W)
    x = source.unsqueeze(0).unsqueeze(0)
    w = kernel.unsqueeze(0).unsqueeze(0)

    pad_d = kernel.shape[0] // 2
    pad_h = kernel.shape[1] // 2
    pad_w = kernel.shape[2] // 2

    # порядок pad: (W_left, W_right, H_left, H_right, D_left, D_right)
    x_padded = F.pad(
        x,
        (pad_w, pad_w,
         pad_h, pad_h,
         pad_d, pad_d)
    )

    y = F.conv3d(x_padded, w)

    y = torch.clamp(y, max=1.0)

    return y.squeeze(0).squeeze(0)
