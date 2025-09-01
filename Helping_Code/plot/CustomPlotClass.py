import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CustomPlotClass:
    def __init__(self):
        pass

    def drawConstraints(self, constraints_data):
        """Check constraints dimensionality and print info."""
        if isinstance(constraints_data, np.ndarray) and constraints_data.ndim == 2 and constraints_data.size > 0:
            numCols = constraints_data.shape[1]
            if numCols == 4:
                print("The array has 4 columns.")
                # Add handling for 2D Hyperrectangle
            elif numCols == 6:
                print("The array has 6 columns.")
                # Add handling for 3D Hyperrectangle
            else:
                print(f"The array has {numCols} columns.")
        else:
            raise ValueError("Input must be a non-empty 2D numpy array.")

    def drawConstraintsIn2D(self, min_max_bounds, title="Rectangular Representation of Constraints", dataPoints2D=None, savePath=None):
        """Draw 2D Hyperrectangles based on min/max bounds."""
        number_of_elements = min_max_bounds.shape[0]
        colors = plt.cm.tab10(np.linspace(0, 1, number_of_elements))

        plt.figure()
        for i in range(number_of_elements):
            rect = min_max_bounds[i, :]
            # rect = [x_min, x_max, y_min, y_max]
            plt.gca().add_patch(
                plt.Rectangle((rect[0], rect[2]), rect[1] - rect[0], rect[3] - rect[2],
                              edgecolor=colors[i], fill=False, linewidth=2)
            )
            if dataPoints2D is not None and len(dataPoints2D) > i:
                plt.scatter(dataPoints2D[i][:, 0], dataPoints2D[i][:, 1])

        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.axis("equal")
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        plt.show()

    def draw2DHyperrectanglesWithVertices(self, verticesCellArray, title="2D Hyperrectangles", color=None, dataPoints2D=None, savePath=None):
        """Draw 2D Hyperrectangles using vertex arrays."""
        plt.figure()
        if color is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(verticesCellArray)))

        for i, vertices in enumerate(verticesCellArray):
            hull = ConvexHull(vertices)
            polygon = np.vstack([vertices[hull.vertices], vertices[hull.vertices[0]]])  # close polygon
            edgeColor = colors[i] if color is None else color
            plt.fill(polygon[:,0], polygon[:,1], alpha=0.3, color=edgeColor)
            plt.plot(polygon[:,0], polygon[:,1], 'k-', linewidth=2)

            if dataPoints2D is not None and len(dataPoints2D) > i:
                plt.scatter(dataPoints2D[i][:,0], dataPoints2D[i][:,1])

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.axis("equal")
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        plt.show()


    def drawDual2DHyperrectanglesWithVertices(self, verticesCellArray1, verticesCellArray2, title="2D Hyperrectangles", color1=None, color2=None, savePath=None):
        """Draw two sets of 2D Hyperrectangles together."""
        plt.figure()
        if color1 is None:
            color1 = np.random.rand(3,)
        if color2 is None:
            color2 = np.random.rand(3,)

        num_colors = len(verticesCellArray1)
        colors = [np.random.rand(3,) for _ in range(num_colors)]

        for v1, v2, c in zip(verticesCellArray1, verticesCellArray2, colors):
            # color1 = np.random.rand(3,)
            # color2 = np.random.rand(3,)

            # c = np.random.rand(3,)


            hull = ConvexHull(v1)
            polygon = np.vstack([v1[hull.vertices], v1[hull.vertices[0]]])
            # c = color1[i] if isinstance(color1, list) else color1
            plt.fill(polygon[:,0], polygon[:,1], alpha=0.3, color=c)
            plt.plot(polygon[:,0], polygon[:,1], color=c, linewidth=1.5)


            # hull1 = ConvexHull(v1)
            # plt.plot(v1[hull1.vertices, 0], v1[hull1.vertices, 1], color=color1, linewidth=2)


            hull2 = ConvexHull(v2)
            polygon2 = np.vstack([v2[hull2.vertices], v2[hull2.vertices[0]]])
            # c = color2[i] if isinstance(color2, list) else color2
            plt.fill(polygon2[:,0], polygon2[:,1], alpha=0.3, color=c)
            plt.plot(polygon2[:,0], polygon2[:,1], color=c, linewidth=1.5)
            # hull2 = ConvexHull(v2)
            # plt.plot(v2[hull2.vertices, 0], v2[hull2.vertices, 1], color=color2, linewidth=2)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.axis("equal")
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        plt.show()



    # def draw3DHyperrectanglesWithVertices(self, vertices, title="3D Hyperrectangles", color=None):
    #     """Draw 3D Hyperrectangles using vertex arrays."""
    #     if not isinstance(vertices, list) or len(vertices) == 0:
    #         raise ValueError("Input must be a non-empty list of vertex arrays.")

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")

    #     if color is None:
    #         colors = plt.cm.tab10(np.linspace(0, 1, len(vertices)))

    #     for i, verts in enumerate(vertices):
    #         hull = ConvexHull(verts)
    #         faces = [verts[simplex] for simplex in hull.simplices]  # use all simplices for faces
    #         faceColor = colors[i] if color is None else color
    #         ax.add_collection3d(Poly3DCollection(faces, facecolors=faceColor, linewidths=1, edgecolors='k', alpha=0.3))

    #     ax.scatter(np.vstack(vertices)[:,0], np.vstack(vertices)[:,1], np.vstack(vertices)[:,2], color='k')
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title(title)
    #     plt.show()


    

    # def drawDual3DHyperrectanglesWithVertices(self, vertices1, vertices2, title="3D Hyperrectangles", color1=None, color2=None):
    #     """Draw two sets of 3D Hyperrectangles together."""
    #     if not isinstance(vertices1, list) or len(vertices1) == 0:
    #         raise ValueError("Input must be a non-empty list of vertex arrays.")
    #     if not isinstance(vertices2, list) or len(vertices2) == 0:
    #         raise ValueError("Input must be a non-empty list of vertex arrays.")

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")

    #     if color1 is None:
    #         color1 = np.random.rand(3,)
    #     if color2 is None:
    #         color2 = np.random.rand(3,)

    #     for verts in vertices1:
    #         hull = ConvexHull(verts)
    #         faces = [verts[simplex] for simplex in hull.simplices]
    #         ax.add_collection3d(Poly3DCollection(faces, facecolors=color1, linewidths=1, edgecolors='k', alpha=0.3))

    #     for verts in vertices2:
    #         hull = ConvexHull(verts)
    #         faces = [verts[simplex] for simplex in hull.simplices]
    #         ax.add_collection3d(Poly3DCollection(faces, facecolors=color2, linewidths=1, edgecolors='k', alpha=0.3))

    #     ax.scatter(np.vstack(vertices1 + vertices2)[:,0], np.vstack(vertices1 + vertices2)[:,1],
    #                np.vstack(vertices1 + vertices2)[:,2], color='k')
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title(title)
    #     plt.show()


    
    def draw3DHyperrectanglesWithVertices(self, vertices, title="3D Hyperrectangles",
                                          color=None, interactive=False, savePath=None):
        """Draw 3D Hyperrectangles with optional interactivity."""
        if not isinstance(vertices, list) or len(vertices) == 0:
            raise ValueError("Input must be a non-empty list of vertex arrays.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if color is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(vertices)))

        for i, verts in enumerate(vertices):
            hull = ConvexHull(verts)
            faces = [verts[simplex] for simplex in hull.simplices]
            faceColor = colors[i] if color is None else color
            ax.add_collection3d(
                Poly3DCollection(faces, facecolors=faceColor,
                                 linewidths=1, edgecolors='k', alpha=0.3)
            )

        ax.scatter(np.vstack(vertices)[:, 0],
                   np.vstack(vertices)[:, 1],
                   np.vstack(vertices)[:, 2], color='k')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        # Handle interactivity
        if interactive:
            ax.view_init(elev=20, azim=30)  # initial camera angle
            plt.ion()
        else:
            plt.ioff()
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        plt.show(block=True)

    def drawDual3DHyperrectanglesWithVertices(self, vertices1, vertices2,
                                              title="Dual 3D Hyperrectangles",
                                              color1=None, color2=None,
                                              interactive=False, savePath=None):
        """Draw two sets of 3D Hyperrectangles together with optional interactivity."""
        if not isinstance(vertices1, list) or len(vertices1) == 0:
            raise ValueError("Input must be a non-empty list of vertex arrays.")
        if not isinstance(vertices2, list) or len(vertices2) == 0:
            raise ValueError("Input must be a non-empty list of vertex arrays.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if color1 is None:
            color1 = np.random.rand(3,)
        if color2 is None:
            color2 = np.random.rand(3,)

        # First set
        for verts in vertices1:
            hull = ConvexHull(verts)
            faces = [verts[simplex] for simplex in hull.simplices]
            ax.add_collection3d(
                Poly3DCollection(faces, facecolors=color1,
                                 linewidths=1, edgecolors='k', alpha=0.3)
            )

        # Second set
        for verts in vertices2:
            hull = ConvexHull(verts)
            faces = [verts[simplex] for simplex in hull.simplices]
            ax.add_collection3d(
                Poly3DCollection(faces, facecolors=color2,
                                 linewidths=1, edgecolors='k', alpha=0.3)
            )

        # Scatter points
        ax.scatter(np.vstack(vertices1 + vertices2)[:, 0],
                   np.vstack(vertices1 + vertices2)[:, 1],
                   np.vstack(vertices1 + vertices2)[:, 2], color='k')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        # Handle interactivity
        if interactive:
            ax.view_init(elev=20, azim=30)
            plt.ion()
        else:
            plt.ioff()

        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        plt.show(block=True)