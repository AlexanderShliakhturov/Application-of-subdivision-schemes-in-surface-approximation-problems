import numpy as np
from tqdm import tqdm

def make_conv_3d(source, kernel):

    result = np.zeros(shape = source.shape)

    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    pad_length = kernel.shape[2] // 2
    
    source_pad = np.pad(source, 
                        pad_width=((pad_height, pad_height), (pad_width, pad_width), (pad_length, pad_length)), 
                        mode='constant', 
                        constant_values=0)
                        

    source_pad_height, source_pad_widht, source_pad_length = source_pad.shape

    kernel_height, kernel_widht, kernel_length = kernel.shape

    for i in tqdm(range(source_pad_height - 2*pad_height)):
        for j in (range(source_pad_widht - 2*pad_width)):
            for k in (range(source_pad_length - 2*pad_length)):

                value = min(
                            np.sum(source_pad[i:i + kernel_height, j: j + kernel_widht, k: k + kernel_length]*kernel),
                            1
                            )
                # print(i,j, k, value)
                result[i][j][k] = value


    return result