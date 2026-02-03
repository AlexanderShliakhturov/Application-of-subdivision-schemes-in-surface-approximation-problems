import numpy as np
import pandas as pd
# import torch
# import scipy
import matplotlib.pyplot as plt
import seaborn

import plotly.graph_objects as go
import plotly.express as px
import torch


def make_html_visual(tensor_3d, name, colorscale='Jet'):

    points = torch.argwhere(tensor_3d != 0)
    
    # print(points)
        
    title = "Облако точек"
        
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    colors = y

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            colorscale=colorscale,
            opacity=0.8,
            colorbar=dict(title="Z"),
        ),
        text=[f'({row[0]}, {row[1]}, {row[2]})' for row in points],
        hovertemplate='<b>Координаты</b>: %{text}<br>' +
                    '<b>Z</b>: %{z}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data' 
        ),
        width=1000,
        height=800,
        hovermode='closest'
    )

    fig.write_html(f"./htmls/{name}.html", include_plotlyjs="cdn", full_html=False)
    print(f"Файл сохранен в ./htmls/{name}.html")
    


