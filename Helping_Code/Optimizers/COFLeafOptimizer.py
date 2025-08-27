import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError
# from sklearn.metrics import mean_squared_error

from Helping_Code.HelpingFunctions import get_leaf_samples
# ---------- Import your real optimizer functions ----------
# Adjust the import path to where QuadraticConstraintModel.py lives in your project.
# If it's in the same package, use a relative import. Here I try absolute then fallback.
try:
    # try direct module import (common when running as script or proper package)
    # from ...QuadraticConstraintModel import constrained_optimization, constrained_optimization_gurobi
    from QuadraticConstraintModel import constrained_optimization, constrained_optimization_gurobi
    _HAS_REAL_OPT = True
except Exception as e:
    # If import fails, provide a helpful message but still allow running a fallback for testing
    _HAS_REAL_OPT = False
    constrained_optimization = None
    constrained_optimization_gurobi = None
    _IMPORT_EXCEPTION = e

    # You can decide whether to fail fast or allow fallback; here we allow fallback but warn.
    import warnings
    warnings.warn(
        "Could not import real optimizers from QuadraticConstraintModel. "
        "Falling back to a safe least-squares local optimizer for testing. "
        f"Import error was: {_IMPORT_EXCEPTION}"
    )

# ---------- Fallback (safe) optimizer if real ones not importable ----------
def _fallback_optimizer(X_leaf, y_leaf, h_max=2.0):
    """
    Simple fallback: least-squares W (n_features x n_outputs), then produce M=(n_outputs,n_features), m0 (n_outputs,)
    This keeps behavior stable for development if CVXPY/Gurobi module is missing.
    """
    X = np.asarray(X_leaf)
    y = np.asarray(y_leaf)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    # Solve X W = y  (W: n_features x n_outputs)
    W, *_ = np.linalg.lstsq(X, y, rcond=None)
    M = W.T  # shape (n_outputs, n_features)
    m0 = (y.mean(axis=0) - X.mean(axis=0) @ W).ravel()
    pred = X @ W + m0
    residuals = y - pred
    h_val = float(np.sum(residuals**2))
    h_val = min(h_val, h_max)
    return M, m0, h_val

# If the real optimizers exist, use them; otherwise point to safe fallback
if not _HAS_REAL_OPT:
    constrained_optimization = _fallback_optimizer
    constrained_optimization_gurobi = _fallback_optimizer



# ---------- dataclass for leaf ----------
@dataclass
class LeafModel:
    leaf_id: int
    M: np.ndarray        # shape (n_outputs, n_features)
    m0: np.ndarray       # shape (n_outputs,)
    h: float
    no_samples: int
    indices: np.ndarray
    bounds: dict         # original-domain bounds

# ---------- COFLeafOptimizer (uses scaler_X to compute original-domain bounds) ----------
'''
class COFLeafOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, tree=None, optimizer="default", h_max=2.0, random_state=None, scaler_X=None):
        self.tree = tree
        self.optimizer = optimizer
        self.h_max = float(h_max)
        self.random_state = check_random_state(random_state)
        self.scaler_X = scaler_X
        self.leaf_models = {}

    def fit(self, X_scaled, y_scaled):
        if self.tree is None:
            raise ValueError("COFLeafOptimizer requires 'tree' attribute to be set before fit().")

        X_scaled = np.asarray(X_scaled)
        y_scaled = np.asarray(y_scaled)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        leaf_samples = get_leaf_samples(self.tree, X_scaled)
        self.leaf_models = {}

        for leaf_id, indices in leaf_samples.items():
            X_leaf = X_scaled[indices]
            y_leaf = y_scaled[indices]

            # choose optimizer (real or fallback)
            if self.optimizer == "gurobi":
                # If real function exists it will be used; else fallback will run
                try:
                    M, m0, h = constrained_optimization_gurobi(X_leaf, y_leaf)
                except Exception as ex:
                    # If gurobi function exists but errors, fallback safely but warn
                    import warnings
                    warnings.warn(f"Gurobi optimizer failed for leaf {leaf_id}: {ex}. Using fallback.")
                    M, m0, h = _fallback_optimizer(X_leaf, y_leaf, h_max=self.h_max)
            else:
                try:
                    M, m0, h = constrained_optimization(X_leaf, y_leaf)
                except Exception as ex:
                    import warnings
                    warnings.warn(f"Optimizer failed for leaf {leaf_id}: {ex}. Using fallback.")
                    M, m0, h = _fallback_optimizer(X_leaf, y_leaf)

            # Defensive handling: if optimizer returned None, use fallback
            # if M is None or m0 is None:
            #     M, m0, h = _fallback_optimizer(X_leaf, y_leaf, h_max=self.h_max)

            M = np.asarray(M)
            m0 = np.asarray(m0).ravel()

            # Compute bounds in original domain (if scaler_X available)
            if self.scaler_X is not None:
                try:
                    X_leaf_orig = self.scaler_X.inverse_transform(X_leaf)
                except Exception:
                    # If inverse_transform fails for whatever reason, fallback to X_leaf
                    X_leaf_orig = X_leaf
            else:
                X_leaf_orig = X_leaf

            bounds = {
                f"feature_{i}": (float(X_leaf_orig[:, i].min()), float(X_leaf_orig[:, i].max()))
                for i in range(X_leaf_orig.shape[1])
            }

            leaf_model = LeafModel(
                leaf_id=int(leaf_id),
                M=M,
                m0=m0,
                h=float(h),
                no_samples=int(len(indices)),
                indices=indices,
                bounds=bounds
            )
            self.leaf_models[int(leaf_id)] = leaf_model

        return self

    def predict(self, X_scaled):
        if not self.leaf_models:
            raise NotFittedError("COFLeafOptimizer is not fitted. Call fit() first.")

        X_scaled = np.asarray(X_scaled)
        n_samples = X_scaled.shape[0]
        sample_leaf = next(iter(self.leaf_models.values()))
        n_outputs = int(sample_leaf.m0.shape[0])

        y_pred = np.zeros((n_samples, n_outputs))

        leaf_samples = get_leaf_samples(self.tree, X_scaled)
        for leaf_id, indices in leaf_samples.items():
            leaf = self.leaf_models[int(leaf_id)]
            # M: (n_outputs, n_features); prediction uses X @ M.T
            y_pred[indices] = X_scaled[indices] @ leaf.M.T + leaf.m0

        return y_pred

'''




from joblib import Parallel, delayed


class COFLeafOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, tree=None, optimizer="default", h_max=2.0, random_state=None, scaler_X=None, n_jobs=-1, verbose=1):
        """
        n_jobs : int
            Number of parallel jobs (default=-1 means all cores).
        verbose : int
            If >0, shows a progress bar.
        """
        self.tree = tree
        self.optimizer = optimizer
        self.h_max = float(h_max)
        self.random_state = check_random_state(random_state)
        self.scaler_X = scaler_X
        self.leaf_models = {}
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit_leaf(self, leaf_id, indices, X_scaled, y_scaled):
        """Helper to optimize one leaf."""
        X_leaf = X_scaled[indices]
        y_leaf = y_scaled[indices]

        print(f"Optimization on Leaf ID = {leaf_id}")
        # choose optimizer (real or fallback)
        if self.optimizer == "gurobi":
            try:
                M, m0, h = constrained_optimization_gurobi(X_leaf, y_leaf)
            except Exception as ex:
                import warnings
                warnings.warn(f"Gurobi optimizer failed for leaf {leaf_id}: {ex}. Using fallback.")
                M, m0, h = _fallback_optimizer(X_leaf, y_leaf, h_max=self.h_max)
        else:
            try:
                M, m0, h = constrained_optimization(X_leaf, y_leaf)
            except Exception as ex:
                import warnings
                warnings.warn(f"Optimizer failed for leaf {leaf_id}: {ex}. Using fallback.")
                M, m0, h = _fallback_optimizer(X_leaf, y_leaf, h_max=self.h_max)

        M = np.asarray(M)
        m0 = np.asarray(m0).ravel()

        # Compute bounds in original domain
        if self.scaler_X is not None:
            try:
                X_leaf_orig = self.scaler_X.inverse_transform(X_leaf)
            except Exception:
                X_leaf_orig = X_leaf
        else:
            X_leaf_orig = X_leaf

        bounds = {
            f"feature_{i}": (float(X_leaf_orig[:, i].min()), float(X_leaf_orig[:, i].max()))
            for i in range(X_leaf_orig.shape[1])
        }

        return LeafModel(
            leaf_id=int(leaf_id),
            M=M,
            m0=m0,
            h=float(h),
            no_samples=int(len(indices)),
            indices=indices,
            bounds=bounds
        )

    def fit(self, X_scaled, y_scaled):
        if self.tree is None:
            raise ValueError("COFLeafOptimizer requires 'tree' to be set before fit().")

        X_scaled = np.asarray(X_scaled)
        y_scaled = np.asarray(y_scaled)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        leaf_samples = get_leaf_samples(self.tree, X_scaled)
        # print("Before Parallel processing in COFLeafOptimizer")
        # Parallel execution with progress bar
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_leaf)(leaf_id, indices, X_scaled, y_scaled)
            for leaf_id, indices in leaf_samples.items()
        )
        # print("After Parallel processing in COFLeafOptimizer")
        # Collect results
        self.leaf_models = {leaf.leaf_id: leaf for leaf in results}
        return self

    def predict(self, X_scaled):
        if not self.leaf_models:
            raise NotFittedError("COFLeafOptimizer is not fitted. Call fit() first.")

        X_scaled = np.asarray(X_scaled)
        n_samples = X_scaled.shape[0]
        sample_leaf = next(iter(self.leaf_models.values()))
        n_outputs = int(sample_leaf.m0.shape[0])

        y_pred = np.zeros((n_samples, n_outputs))

        leaf_samples = get_leaf_samples(self.tree, X_scaled)
        for leaf_id, indices in leaf_samples.items():
            leaf = self.leaf_models[int(leaf_id)]
            # M: (n_outputs, n_features); prediction uses X @ M.T
            y_pred[indices] = X_scaled[indices] @ leaf.M.T + leaf.m0

        return y_pred