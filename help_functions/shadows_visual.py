import torch
from skimage import measure
import numpy as np
import pyvista as pv



def volume_to_mesh(volume: torch.Tensor, level=0.5):
    volume_np = volume.detach().cpu().numpy().astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume_np,
        level=level
    )

    return verts, faces, normals


def shadows_visual(input_tensor: torch.Tensor, name: str):

    surface = input_tensor

    surface_rot = torch.rot90(surface, k=1, dims=(1,2))
    surface_rot = torch.rot90(surface_rot, k=2, dims=(0,1))
    surface_rot = torch.rot90(surface_rot, k=-1, dims=(0,1))


    verts, faces, normals = volume_to_mesh(surface_rot)

    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]
    ).astype(np.int64)

    mesh = pv.PolyData(verts, faces_pv)
    mesh.point_data["Normals"] = normals

    plotter = pv.Plotter(off_screen=True,
                        window_size=(3000, 3000))
    plotter.add_mesh(
        mesh,
        color="gold",
        smooth_shading=False,
        specular=0.5,
        specular_power=30
    )
    plotter.view_yz()



    # plotter.show(screenshot="figure_x0.png",
    #              window_size=(3000, 3000))
    plotter.save_graphic(f"{name}.png")
    plotter.close()

# plotter.save_graphic
