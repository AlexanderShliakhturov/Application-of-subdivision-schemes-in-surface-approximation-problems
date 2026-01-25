import numpy as np

def make_conv_2d(source, kernel):

    result = np.zeros(shape = source.shape)

    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    
    source_pad = np.pad(source, 
                        pad_width=((pad_height, pad_height), (pad_width, pad_width)), 
                        mode='constant', 
                        constant_values=0)
    
    # print(source_pad)
                        
    source_pad_height, source_pad_widht = source_pad.shape

    kernel_height, kernel_widht = kernel.shape

    for i in range(source_pad_height - 2*pad_width):
        for j in range(source_pad_widht - 2*pad_width):
            value = min(
                        np.sum(source_pad[i:i + kernel_height, j: j + kernel_widht]*kernel),
                        1
                        )
            
            # value = np.sum(source_pad[i:i + kernel_height, j: j + kernel_widht]*kernel)
            # print(i,j, value)
            result[i][j] = value


    return result
