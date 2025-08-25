import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_rectangle(rect: "Rectangle", ax=None, color="blue", alpha=0.3):
    """Plot Rectangle (2D → filled box, >=3D → interactive cuboid)."""
    if rect.dimension == 2:
        if ax is None:
            fig, ax = plt.subplots()
        width = rect.upper_bounds[0] - rect.lower_bounds[0]
        height = rect.upper_bounds[1] - rect.lower_bounds[1]
        ax.add_patch(
            patches.Rectangle(
                (rect.lower_bounds[0], rect.lower_bounds[1]),
                width, height,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=alpha
            )
        )
        ax.set_aspect("equal")
        return ax

    elif rect.dimension >= 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ax is None:
            plt.ion()  # Interactive mode
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        x = [rect.lower_bounds[0], rect.upper_bounds[0]]
        y = [rect.lower_bounds[1], rect.upper_bounds[1]]
        z = [rect.lower_bounds[2], rect.upper_bounds[2]]

        corners = np.array([[i, j, k] for i in x for j in y for k in z])

        faces = [
            [corners[j] for j in [0, 1, 3, 2]],
            [corners[j] for j in [4, 5, 7, 6]],
            [corners[j] for j in [0, 1, 5, 4]],
            [corners[j] for j in [2, 3, 7, 6]],
            [corners[j] for j in [0, 2, 6, 4]],
            [corners[j] for j in [1, 3, 7, 5]],
        ]

        ax.add_collection3d(
            Poly3DCollection(faces, alpha=alpha, facecolor=color, linewidths=1, edgecolor="k")
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return ax

    else:
        raise ValueError("Rectangle plotting is only supported for 2D or >=3D")


def plot_sphere(sphere: "Sphere", ax=None, color="red", alpha=0.3):
    """Plot Sphere (2D → circle, >=3D → interactive 3D ball)."""
    if sphere.dimension == 2:
        if ax is None:
            fig, ax = plt.subplots()
        circle = patches.Circle(
            (sphere.centre[0], sphere.centre[1]),
            sphere.radius,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha
        )
        ax.add_patch(circle)
        ax.set_aspect("equal")
        return ax

    elif sphere.dimension >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        if ax is None:
            plt.ion()  # interactive mode
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = sphere.centre[0] + sphere.radius * np.cos(u) * np.sin(v)
        y = sphere.centre[1] + sphere.radius * np.sin(u) * np.sin(v)
        z = sphere.centre[2] + sphere.radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return ax

    else:
        raise ValueError("Sphere plotting is only supported for 2D or >=3D")
