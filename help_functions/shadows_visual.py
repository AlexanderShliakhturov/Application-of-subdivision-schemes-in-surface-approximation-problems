import torch
import os
from skimage import measure
import numpy as np
import pyvista as pv
import plotly.graph_objects as go



def volume_to_mesh(volume: torch.Tensor, level=0.5):
    volume_np = volume.detach().cpu().numpy().astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume_np,
        level=level,
        allow_degenerate=False,
        gradient_direction='descent',
        step_size=1
        
    )

    return verts, faces, normals


def shadows_visual(
    input_tensor: torch.Tensor,
    name: str,
    subfolder: str,
    layers=None,
    level=0.5,
    ambient=0.3,
    diffuse=0.8,
    specular=0.5,
    specular_power=30,
    roughness=0.3,
    window_size=(1000, 1000),
    camera_position=(50,850,500),
    light_position=(50,850,500)
):

    surface = input_tensor

    if layers == 'horizontal':
        surface_rot = torch.rot90(surface, k=1, dims=(1,2))
        surface_rot = torch.rot90(surface_rot, k=1, dims=(0,1))
        surface_rot = torch.rot90(surface_rot, k=-1, dims=(0,1))
    elif layers == 'vertical':
        surface_rot = torch.rot90(surface, k=1, dims=(0,1))
    else:
        surface_rot = surface

    verts, faces, normals = volume_to_mesh(surface_rot, level=level)

    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]
    ).astype(np.int64)

    mesh = pv.PolyData(verts, faces_pv)
    mesh.point_data["Normals"] = normals

    plotter = pv.Plotter(window_size=window_size)

    plotter.add_mesh(
        mesh,
        color="gold",
        pbr=True,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        specular_power=specular_power,
        roughness=roughness,
    )
    plotter.add_axes()
    
    camera_position=camera_position
    light_position=light_position
    
    plotter.add_light(pv.Light(position=light_position, intensity=2))
    
    plotter.camera_position = camera_position
    root_folder = 'subdivision_results'
    os.makedirs(f"./{root_folder}/{subfolder}", exist_ok=True)
    filepath = os.path.join(root_folder, subfolder, f"{name}.png")
    plotter.show(screenshot=filepath)
    print(f"Изображение сохранено в {filepath}")
    
    
def points_visual(
    input_tensor: torch.Tensor,
    name: str,
    subfolder: str,
    threshold=0.5,
    layers=None,
    point_size=5,
    window_size=(1000, 1000),
    camera_position=(50, 850, 500)
):
    surface = input_tensor

    if layers == 'horizontal':
        surface_rot = torch.rot90(surface, k=1, dims=(1, 2))
        surface_rot = torch.rot90(surface_rot, k=1, dims=(0, 1))
        surface_rot = torch.rot90(surface_rot, k=-1, dims=(0, 1))
    elif layers == 'vertical':
        surface_rot = torch.rot90(surface, k=1, dims=(0, 1))
    else:
        surface_rot = surface

    volume_np = surface_rot.detach().cpu().numpy()

    points = np.argwhere(volume_np >= threshold)

    cloud = pv.PolyData(points)

    plotter = pv.Plotter(window_size=window_size)

    plotter.add_mesh(
        cloud,
        color="gold",
        render_points_as_spheres=True,
        point_size=point_size
    )

    plotter.add_axes()
    plotter.camera_position = camera_position

    root_folder = 'subdivision_results'
    os.makedirs(f"./{root_folder}/{subfolder}", exist_ok=True)
    filepath = os.path.join(root_folder, subfolder, f"{name}.png")
    plotter.show(screenshot=filepath)
    print(f"Изображение сохранено в {filepath}")

# plotter.save_graphic

# def shadows_visual_UI(
#     input_tensor: torch.Tensor,
#     name: str,
#     layers = 'vertical',
#     isomin=0.6,
#     isomax=1.5,
#     downsample=1,
#     colorscale=[[0, 'rgb(255,255,0)'], [1, 'rgb(255,255,0)']]
# ):
#     """
#     Воксельная визуализация через Plotly Isosurface
#     с тенями и сохранением в PNG.
#     """

#     surface = input_tensor
    
#     if layers == 'horizontal':

#         surface_rot = torch.rot90(surface, k=1, dims=(1,2))
#         surface_rot = torch.rot90(surface_rot, k=2, dims=(0,1))
#         surface_rot = torch.rot90(surface_rot, k=-1, dims=(0,1))

#     elif layers == 'vertical':
#         surface_rot = torch.rot90(surface, k=1, dims=(0,1))
    
#     else:
#         print('Wrong "layers" variable')
#         return

#     volume = surface_rot.detach().cpu().numpy()

#     D, H, W = volume.shape

#     z, y, x = np.mgrid[
#         0:D:downsample,
#         0:H:downsample,
#         0:W:downsample
#     ]

#     values = volume[::downsample, ::downsample, ::downsample]

#     fig = go.Figure(
#         data=go.Isosurface(
#             x=x.flatten(),
#             y=y.flatten(),
#             z=z.flatten(),
#             value=values.flatten(),

#             isomin=isomin,
#             isomax=isomax,

#             surface_count=1,
#             colorscale=colorscale,
#             showscale=False,

#             caps=dict(x_show=False, y_show=False, z_show=False),

#             lighting=dict(
#                 ambient=0.3,
#                 diffuse=0.8,
#                 specular=0.5,
#                 roughness=0.3,
#                 fresnel=0.2
#             ),

#             lightposition=dict(
#                 x=200,
#                 y=200,
#                 z=300
#             ),
#         )
#     )

#     # убираем оси
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(visible=False),
#             yaxis=dict(visible=False),
#             zaxis=dict(visible=False),
#             bgcolor="white"
#         ),
#         margin=dict(l=0, r=0, t=0, b=0)
#     )

#     # сохраняем PNG
#     # fig.write_image(
#     #     f"{name}.png",
#     #     width=3000,
#     #     height=3000,
#     #     scale=1
#     # )

#     fig.write_html(f"{name}.html")
