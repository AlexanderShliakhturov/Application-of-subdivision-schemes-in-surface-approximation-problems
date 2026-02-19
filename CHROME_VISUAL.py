import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from numpy import loadtxt
import plotly.graph_objects as go
import plotly.io as pio
import torch
from time import time
pio.renderers.default='browser'

volume = torch.load("./tensor_files/x0_clone_filtered_ceil.pt", weights_only=True)

volume = torch.rot90(volume, k=1, dims=(0,1))

downsample = 3
# "green"
colorscale = [[0, 'rgb(0,255,0)'], [1, 'rgb(0,255,0)']]
# "yellow":
colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,255,0)']]
D, H, W = volume.shape

# --- координатная сетка
z, y, x = np.mgrid[
    0:D:downsample,
    0:H:downsample,
    0:W:downsample
]
values = volume[::downsample, ::downsample, ::downsample]

fig = go.Figure(data = go.Isosurface(x=x.flatten(),
                                    y=y.flatten(),
                                    z=z.flatten(),
                                    value=values.flatten(),
                                    colorscale = colorscale, #colorscale = 'rdbu', #'plotly3', #'matter', #'oranges', #'edge',
                                    isomin = 0.7,
                                    isomax = 1.5,
                                    showscale = True,
                                    caps = dict(x_show = False, y_show = False)))

fig.update_layout(scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis =dict(visible=False)))
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()