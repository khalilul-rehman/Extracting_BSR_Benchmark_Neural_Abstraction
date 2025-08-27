import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

from QuadraticConstraintModel import train_COF_on_leaves, get_h_from_COF

from QuadraticConstraintModel import get_feature_bounds_from_COF, predict_from_COF, get_elevated_vertices

from Helping_Code.HelpingFunctions import load_dataset, normalized_root_mean_square_error
from Helping_Code.CustomHyperrectangle import CustomHyperrectangle
from Helping_Code.plot.CustomPlotClass import CustomPlotClass

from COFPipeline import COFPipeline


data_Directory = "Dataset/"

X,y = load_dataset(data_Directory + "SteamGovernor/data_SteamGovernor_10000000.csv",num_attributes = 3, num_classes = 3 )
print(f"Size of the Data Set\n Shape of X = {X.shape} \n Shape of y = {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.1)
print(f" Shape of X_Training = {X_train.shape} \n Shape of X_Testing = {X_test.shape}")

# X_train = X
# y_train = y
# 'max_depth': 5,
tree_params = { 'min_samples_leaf': 1000000}
pipeline = COFPipeline(optimizer="gurobi", tree_params=tree_params, scale=True, poly_degree=2, n_jobs = 2)
#pipeline = COFPipeline(optimizer="gurobi", max_depth=5, scale=True,auto_tune_poly=True, max_poly_degree=3, n_jobs = 10)
pipeline.fit(X_train, y_train)

# Inspect some leaf bounds (original domain)
cof_model = pipeline.cof_model
print("Number of modeled leaves:", len(cof_model.leaf_models))
for lid, leaf in list(cof_model.leaf_models.items()):
    print("H = ", leaf.h)
    # print(f"leaf {lid}: samples={leaf.no_samples}, h={leaf.h:.6g}, bounds={leaf.bounds}")



y_pred = pipeline.predict(X_test)

print(y_test[:10])
print(y_pred[:10])


'''
# Use 'default' (cvx or fallback) or 'gurobi' (if you have gurobipy)
pipeline = COFPipeline(optimizer="gurobi", max_depth=4, scale=True, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = normalized_root_mean_square_error(y_test, y_pred)
print("MSE per output:", mse)

# Inspect some leaf bounds (original domain)
cof_model = pipeline.cof_model
print("Number of modeled leaves:", len(cof_model.leaf_models))
for lid, leaf in list(cof_model.leaf_models.items()):
    print("H = ", leaf.h)
    # print(f"leaf {lid}: samples={leaf.no_samples}, h={leaf.h:.6g}, bounds={leaf.bounds}")

'''
    

'''
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

numLeaves = tree.get_n_leaves()
print(f"Number of leaves: {tree.get_n_leaves()}")
print(f"Total depth of tree: {tree.get_depth()}")
print(f"Number of nodes: {tree.tree_.node_count}")


print("NRMSE [Tree Based] of Training = ", normalized_root_mean_square_error(y_train, tree.predict(X_train)))
print("NRMSE [Tree Based] of Testing  = ", normalized_root_mean_square_error(y_test, tree.predict(X_test)))

COF_model_tree = train_COF_on_leaves(X_train, y_train, tree, optimizer="gurobi")

high_h = get_h_from_COF(COF_model_tree, greater_then= -np.inf)
print("following are high h", high_h)

constraints_on_leaves = get_feature_bounds_from_COF(COF_model_tree)

customHyperrectangle = CustomHyperrectangle()

vertices_of_hyperrectangle = customHyperrectangle.getVerticesFromBounds(constraints_on_leaves)



customPlot = CustomPlotClass()

num_cols = constraints_on_leaves.shape[1]

if num_cols == 4:
    # 2D Hyperrectangles
    # customPlot.draw2DHyperrectanglesWithVertices(
    #     vertices_of_polytopes,
    #     title='Constraints as 2D-Rectangles',
    #     dataPoints2D=trajectories_on_leaves
    # )
    customPlot.draw2DHyperrectanglesWithVertices(
        vertices_of_hyperrectangle,
        color='#8f9119'
    )

elif num_cols == 6:
    # 3D Hyperrectangles
    customPlot.draw3DHyperrectanglesWithVertices(
        vertices_of_hyperrectangle,
        interactive=True
    )
else:
    # More than 3D, project into 2D for plotting
    customPlot.drawConstraintsIn2D(
        constraints_on_leaves,
        title='More than 3D Constraints in 2D'
    )

# Compute min and max along each dimension
minOfAllDimensions = np.min(constraints_on_leaves, axis=0)
maxOfAllDimensions = np.max(constraints_on_leaves, axis=0)

elevated_vertices = get_elevated_vertices(COF_model_tree, vertices_of_hyperrectangle)



# Check the number of rows in the first element of vertices_of_elevated_polytopes_before_span
dim = elevated_vertices[0].shape[0]

if dim == 4:
    # 2D polytopes
    customPlot.drawDual2DHyperrectanglesWithVertices(
        vertices_of_hyperrectangle,
        elevated_vertices,
        color1='#8f9119',
        color2='#197b91',
        title='Elevated 2D-Polytopes'
    )
elif dim == 8:
    # 3D polytopes
    customPlot.drawDual3DHyperrectanglesWithVertices(
        vertices_of_hyperrectangle,
        elevated_vertices,
        color1='#8f9119',
        color2='#197b91',
        title='Elevated 3D-Polytopes',
        interactive=True
    )
else:
    # Fallback for other cases
    customPlot.drawDual2DHyperrectanglesWithVertices(
        vertices_of_hyperrectangle,
        elevated_vertices,
        title='More than 3D Polytopes in 2D'
    )


customPlot.drawDual3DHyperrectanglesWithVertices(
        [vertices_of_hyperrectangle[0]],
        [elevated_vertices[0]],
        color1='#8f9119',
        color2='#197b91',
        title='Elevated 3D-Polytopes',
        interactive=True
    )

p1, p2, min_distance = customHyperrectangle.minimize_hyperrectangle_distance_dual(elevated_vertices[0], vertices_of_hyperrectangle[0])
print(p1,p2,min_distance)



# Get closest points and distance
X_opt, Xp_opt, min_distance = customHyperrectangle.minimize_hyperrectangle_distance_dual(
    elevated_vertices[0], vertices_of_hyperrectangle[0]
)

# Determine dimension
dim = elevated_vertices[0].shape[1]

if dim == 2:
    # 2D plotting
    plt.figure()
    
    def plot_2d_hyperrectangle(vertices, color='cyan', alpha=0.3):
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        plt.fill(vertices[hull.vertices, 0], vertices[hull.vertices, 1], color=color, alpha=alpha)
        plt.plot(vertices[hull.vertices, 0], vertices[hull.vertices, 1], 'k-', linewidth=1)
    
    # Plot rectangles
    plot_2d_hyperrectangle(elevated_vertices[0], color='cyan', alpha=0.3)
    plot_2d_hyperrectangle(vertices_of_hyperrectangle[0], color='orange', alpha=0.3)
    
    # Plot closest points
    plt.scatter(*X_opt, color='blue', s=50, label='Closest point on elevated rectangle')
    plt.scatter(*Xp_opt, color='red', s=50, label='Closest point on rectangle')
    
    # Draw line connecting closest points
    plt.plot([X_opt[0], Xp_opt[0]], [X_opt[1], Xp_opt[1]], color='green', linewidth=2, label=f'Min distance: {min_distance:.3f}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Hyperrectangle Minimum Distance')
    plt.legend()
    plt.axis('equal')
    plt.show()
    
else:
    # 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def plot_3d_hyperrectangle(ax, verts, color='cyan', alpha=0.3):
        faces = np.array([[0, 1, 3, 2], [0, 1, 5, 4], [0, 2, 6, 4],
                          [1, 3, 7, 5], [2, 3, 7, 6], [4, 5, 7, 6]])
        poly3d = [[verts[idx] for idx in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha))
    
    # Plot hyperrectangles
    plot_3d_hyperrectangle(ax, elevated_vertices[0], color='cyan', alpha=0.3)
    plot_3d_hyperrectangle(ax, vertices_of_hyperrectangle[0], color='orange', alpha=0.3)
    
    # Plot closest points
    ax.scatter(*X_opt, color='blue', s=50, label='Closest point on elevated hyperrectangle')
    ax.scatter(*Xp_opt, color='red', s=50, label='Closest point on hyperrectangle')
    
    # Draw line connecting closest points
    line_pts = np.vstack([X_opt, Xp_opt])
    ax.plot(line_pts[:,0], line_pts[:,1], line_pts[:,2], color='green', linewidth=2, label=f'Min distance: {min_distance:.3f}')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title('3D Hyperrectangle Minimum Distance')
    ax.legend()
    plt.show(block=True)


'''