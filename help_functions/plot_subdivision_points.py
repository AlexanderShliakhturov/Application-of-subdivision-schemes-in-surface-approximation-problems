import matplotlib.pyplot as plt
import torch

def plot_subdivision_points(data: torch.Tensor, 
                            threshold=0.05, 
                            title="", 
                            figsize = (10,10), 
                            do_ceil = False,
                            visual_type = "scatter"):
    """
    data:
        2D: (H, W)
        3D: (D, H, W)
    """
    v = data.detach().cpu()
    
    if do_ceil:
        v = (v > threshold).float()
    
    idx = torch.nonzero(v > threshold)

    if v.ndim == 2:
        y = idx[:, 0].numpy()
        x = idx[:, 1].numpy()
        vals = v[idx[:, 0], idx[:, 1]]

        plt.figure(figsize=figsize)
        
        if visual_type == "scatter":
            plt.scatter(
                x, y,
                c=vals.numpy(),
                cmap="viridis",
                s=10
            )
        
        else:
            plt.imshow(v, 
            cmap='viridis', 
            aspect='auto',
            origin='lower',)
        
        plt.colorbar(label="f value")
        plt.gca().set_aspect("equal")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    elif v.ndim == 3:
        z = idx[:, 0].numpy()
        y = idx[:, 1].numpy()
        x = idx[:, 2].numpy()
        vals = v[idx[:, 0], idx[:, 1], idx[:, 2]]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        p = ax.scatter(
            x, y, z,
            c=vals.numpy(),
            cmap="viridis",
            s=5,
            alpha=0.7
        )

        fig.colorbar(p, ax=ax, shrink=0.6, label="f value")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Поддерживаются только 2D и 3D данные")
