import numpy as np
import pandas as pd
import os
# import torch
# import scipy
import matplotlib.pyplot as plt
import seaborn
import torch

import plotly.graph_objects as go
import plotly.express as px


def make_scatter_visual(tensor_3d, name, subfolder):
    
    plt.ioff()
    
    points = torch.argwhere(tensor_3d != 0)
        
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=45)
    
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
    os.makedirs(f"./{subfolder}/images", exist_ok=True)
    path = f'./{subfolder}/images/{name}.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"Изображение сохранено в {path}")
    print(f"Всего точек: {len(points)}")
    print(f"Размер тензора: {tensor_3d.shape}")
    print(f"Плотность: {len(points) / np.prod(tensor_3d.shape):.6f}")
    
    
    
def set_axes_equal(ax, x, y, z, zoom=0.7):
    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ) * zoom

    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)


def make_scatter_with_plane(tensor_3d, path, slice_idx, dpi=120):
    points = torch.argwhere(tensor_3d != 0)

    x = points[:, 0].cpu().numpy()
    y = points[:, 1].cpu().numpy()
    z = points[:, 2].cpu().numpy()

    # Разделяем точки относительно секущей плоскости z = slice_idx
    below_plane = z < slice_idx
    on_plane = z == slice_idx
    above_plane = z > slice_idx

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Точки ниже плоскости
    ax.scatter(
        x[below_plane], y[below_plane], z[below_plane],
        color='blue',
        s=5,
        alpha=0.05,
        edgecolors='none',
        # label='Ниже плоскости'
    )

    # Точки выше плоскости — бледнее
    ax.scatter(
        x[above_plane], y[above_plane], z[above_plane],
        color='gray',
        s=1,
        alpha=0.18,
        edgecolors='none',
        # label='Выше плоскости'
    )

    # Размеры тензора
    nx, ny, nz = tensor_3d.shape

    # Увеличенная секущая плоскость XOY на высоте z = slice_idx
    margin = 50

    X_plane, Y_plane = np.meshgrid(
        [-margin, nx - 1 + margin],
        [-margin, ny - 1 + margin]
    )

    Z_plane = np.full_like(X_plane, slice_idx)

    ax.plot_surface(
        X_plane,
        Y_plane,
        Z_plane,
        alpha=0.25,
        color='red',
        edgecolor='black',
        linewidth=1
    )

    # Подсветка точек, которые лежат прямо на плоскости
    ax.scatter(
        x[on_plane], y[on_plane], z[on_plane],
        color='red',
        s=14,
        alpha=1.0,
        edgecolors='black',
        linewidth=0.5,
        # label='Сечение'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.set_title(f'Облако точек и секущая плоскость z = {slice_idx}')

    set_axes_equal(ax, x, y, z, zoom=0.7)

    ax.view_init(elev=25, azim=-45)

    # ax.legend(loc='upper right')

    plt.tight_layout()

    fig.savefig(path, dpi=dpi)
    plt.close(fig)