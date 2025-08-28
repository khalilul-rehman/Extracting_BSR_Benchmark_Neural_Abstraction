import cvxpy as cp
from collections import defaultdict
import numpy as np

import gurobipy as gp
from gurobipy import GRB

#Constrained Optimization Function

def constrained_optimization(X_leaf, y_leaf):
    n_samples, n_features = X_leaf.shape
    n_outputs = y_leaf.shape[1]

    M = cp.Variable((n_outputs, n_features))
    m0 = cp.Variable((n_outputs,))
    h = cp.Variable(nonneg=True)

    prediction = X_leaf @ M.T + m0
    constraint = cp.sum_squares(prediction - y_leaf)
    objective = cp.Minimize(h)
    constraints = [constraint <= h]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    return M.value, m0.value, h.value if h.value is not None else np.inf

def constrained_optimization_gurobi(X_leaf, y_leaf):
    n_samples, n_features = X_leaf.shape
    n_outputs = y_leaf.shape[1]

    # Create model
    model = gp.Model("constrained_optimization")
    model.setParam("OutputFlag", 0)  # silence solver output

    # Decision variables
    M = model.addVars(n_outputs, n_features, lb=-GRB.INFINITY, name="M")
    m0 = model.addVars(n_outputs, lb=-GRB.INFINITY, name="m0")
    h = model.addVar(lb=0, name="h")  # non-negative

    # Compute squared residual sum
    residuals = []
    for i in range(n_samples):
        for k in range(n_outputs):
            expr = gp.LinExpr()
            expr.add(m0[k])
            for j in range(n_features):
                expr.add(M[k, j] * X_leaf[i, j])
            residuals.append((expr - y_leaf[i, k]) * (expr - y_leaf[i, k]))

    # Constraint: sum of squared residuals <= h
    model.addConstr(gp.quicksum(residuals) <= h)

    # Objective: minimize h
    model.setObjective(h, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    # Extract results
    if model.status == GRB.OPTIMAL:
        M_val = np.array([[M[k, j].X for j in range(n_features)] for k in range(n_outputs)])
        m0_val = np.array([m0[k].X for k in range(n_outputs)])
        h_val = h.X
    else:
        M_val, m0_val, h_val = None, None, np.inf

    return M_val, m0_val, h_val





def get_leaf_samples(tree_model, X):
    leaf_indices = tree_model.apply(X)
    leaf_samples = defaultdict(list)
    for i, leaf in enumerate(leaf_indices):
        leaf_samples[leaf].append(i)
    return leaf_samples


def train_COF_on_leaves(X_train, y_train, tree,feature_names=None, optimizer = "gurobi"):
    # optimizer can be { "gurobi" or "CVXPY + SCS"}
    leaf_samples = get_leaf_samples(tree, X_train)
    tree_extracted_info = []
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    for leaf_id, indices in leaf_samples.items():
        X_leaf = X_train[indices]
        y_leaf = y_train[indices]
        if optimizer == "gurobi":
            M, m0, h = constrained_optimization_gurobi(X_leaf, y_leaf)
        else:
            M, m0, h = constrained_optimization(X_leaf, y_leaf)
        model = {} # models = {leaf_Index:{h, M, m0}}
        model["leaf_id"] = leaf_id
        model["CO_Model"] = {'M': M, 'm0': m0, 'h': h }
        model["no_samples"] = len(indices)
        model["indices"] = indices
        
        bounds = {
                feature_names[i]: (X_leaf[:, i].min(), X_leaf[:, i].max())
                for i in range(X_leaf.shape[1])
            }
        model["bounds"] = bounds
        tree_extracted_info.append(model)
    return tree_extracted_info




def get_h_from_COF(COF_model_tree, greater_then = -np.inf):
    high_h = []
    # bounds_of_high_h = []
    # no_samples_of_high_h = []
    for item in COF_model_tree:
        # print(item['CO_Model']['h'])
        if item['CO_Model']['h'] > greater_then:
            high_h.append(item['CO_Model']['h'])
            # bounds_of_high_h.append(item['bounds'])
            # no_samples_of_high_h.append(item['no_samples'])
            # print(item['CO_Model']['h'])
            # print(item['no_samples'])
    return high_h
        


def get_feature_bounds_from_COF(COF_model_tree):
    rows = []
    for item in COF_model_tree:
        # print(item["bounds"])
        row = []
        for feature, (low, high) in item["bounds"].items():
            row.append(float(low))
            row.append(float(high))
        rows.append(row)
    return np.array(rows)




def predict_from_COF(COF_model_tree, X_new, tree):
    """
    Predicts outputs for new samples X_new based on the COF model.

    Parameters:
    -----------
    COF_model_tree : list of dict
        Output from train_COF_on_leaves.
        Each dict contains 'leaf_id', 'CO_Model': {'M','m0','h'}, etc.
    X_new : np.ndarray
        New input samples, shape (n_samples, n_features)
    leaf_indices : np.ndarray or None
        Optional precomputed leaf indices for each sample. 
        If None, function assumes you will map samples to leaves externally.

    Returns:
    --------
    y_pred : np.ndarray
        Predicted outputs, shape (n_samples, n_outputs)
    """
    n_samples = X_new.shape[0]
    n_outputs = next(iter(COF_model_tree))['CO_Model']['M'].shape[0]

    y_pred = np.zeros((n_samples, n_outputs))

    leaf_samples = get_leaf_samples(tree, X_new)

    for leaf_id, indices in leaf_samples.items():
        # X_leaf = X_new[indices]
        leaf_model = leaf_model = next(item for item in COF_model_tree if item['leaf_id'] == leaf_id) 
        M = leaf_model['CO_Model']['M']
        m0 = leaf_model['CO_Model']['m0']
        y_pred[indices] = X_new[indices] @ M.T + m0
    
    

    return y_pred


def get_elevated_vertices(COF_model_tree, vertices):
    elevated_vertices = []
    for idx, item in enumerate(COF_model_tree):
        leaf_model = item['CO_Model']
        M = leaf_model['M']
        m0 = leaf_model['m0']
        elevated_vertices.append(vertices[idx] @ M.T + m0)
    
    return elevated_vertices
            
