import numpy as np
import pandas as pd
# import torch
# import scipy
import matplotlib.pyplot as plt
import seaborn

import plotly.graph_objects as go
import plotly.express as px


def make_scatter_visual(tensor_3d, name):
    
    plt.ioff()
    
    points = np.argwhere(tensor_3d != 0)
        
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x, y, z, 
                        c=z,
                        cmap='viridis',
                        s=1,         
                        alpha=0.7,
                        edgecolors='k',
                        linewidth=0.5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'3D облако точек ({len(points)} точек)', fontsize=14)
    
    
    plt.colorbar(scatter, ax=ax, label='Высота (Z)')
    
    ax.set_box_aspect([1, 1, 1])
    
    # plt.show()
    
    fig.savefig(f'./images/{name}.png')
    
    print(f"Всего точек: {len(points)}")
    print(f"Размер тензора: {tensor_3d.shape}")
    print(f"Плотность: {len(points) / np.prod(tensor_3d.shape):.6f}")