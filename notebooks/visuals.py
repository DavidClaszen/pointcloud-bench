import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # noqa
from typing import Union


def visualize_cloud(
    data: np.array,
    i: int,
    ax=None,
    colorbar: bool = False,
    return_sc: bool = False,
    point_size: int = 1
):
    """Visualizes point cloud data with normals in 2D plot.

    Args:
        data (np.array): (n, points, 6) where n=samples, points=point clouds,
            6 = 3 dimensions + 3 normals
        i (int): n index of sample to visualize
        ax (matplotlib.axes, optional): Supply ax if needed.
        colorbar (bool, optional): Adds a colorbar for normals. Defaults False.
        return_sc (bool, optional): Returns the plot or only visualize it.
            Defaults to False. Used for building plots of multiples.
        point_size (int, optional): Point size of visualized points.
            Defaults to 1.

    Returns:
        mpl_toolkits.mplot3d.art3d.Path3DCollection: 3D Plot object.
    """
    sample = data[i]
    points = sample[:, :3]
    normals = sample[:, 3:]

    created_fig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=normals[:, 2],
        s=point_size,
        cmap='viridis'
    )

    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if colorbar and created_fig:
        plt.colorbar(sc, label='Normal Z')

    if return_sc:
        return sc


def visualize_3d(
    data: np.array,
    i: int
):
    """Visualizes point clouds with normals in 3D plot.

    Args:
        data (np.array): (n, points, 6) where n=samples, points=point clouds,
            6 = 3 dimensions + 3 normals
        i (int): n index of sample to visualize
    """
    sample = data[i]
    points = sample[:, :3]
    normals = sample[:, 3:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    o3d.visualization.draw_geometries([pcd])


def visualize_samples(
    data: Union[list[np.array], np.array],
    title: str,
    indices: list[int] = None,
    figsize: tuple = (12, 8),
    plot_dims: tuple = (2, 3),
    labels: list[str] = None,
    point_size: list[int] = None,
    font_size: int = 8
):
    """Visualize multiple samples using `visualize_cloud`

    Args:
        data (list[np.array]): Either an array (n, points, 6) where n=samples,
            points=point clouds, 6 = 3 dimensions + 3 normals, or a list of
            arrays, with each list item a single row (to accommodate different
            data sizes).
        title (str): Title for the whole plot.
        indices (list[int]): List of integer indices to visualize. If None,
            defaults to integers for 0:len(data).
        figsize (tuple): Tuple of dimensions for figure size, default (12, 8).
        plot_dims (tuple(int)): How many rows and columns of plots to plot,
            defaults to (2, 3).
        labels (list[str], optional): List of titles for each individual plot.
            Defaults to None.
        point_size (list[int], optional): List of sizes of visualized points,
            can be different for each plot. Defaults to 0.8 if None.
        font_size (int): Font size of subplot titles, defaults to 8.
    """
    plt.rcParams.update({'font.size': font_size})
    fig, axs = plt.subplots(
        plot_dims[0],
        plot_dims[1],
        subplot_kw={'projection': '3d'},
        figsize=figsize
    )
    if indices is None:
        indices = range(0, len(data))
    if point_size is None:
        point_size = [0.8 for _ in range(0, len(data))]
    sc = None
    counter = 0
    for ax, idx in zip(axs.ravel(), indices):
        sc = visualize_cloud(
            data=data,
            i=idx,
            ax=ax,
            colorbar=False,
            return_sc=True,
            point_size=point_size[counter]
        )
        if labels is not None:
            ax.set_title(labels[counter])
        else:
            ax.set_title(f'Sample {idx}')
        counter += 1

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
