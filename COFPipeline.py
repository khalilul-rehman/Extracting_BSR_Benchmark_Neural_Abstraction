
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Helping_Code.Optimizers.COFLeafOptimizer import COFLeafOptimizer

class COFPipeline:
    def __init__(self, optimizer="default", tree_params=None, scale=True, random_state=42,
                 poly_degree=1, auto_tune_poly=False, max_poly_degree=3, n_jobs=2):
        self.optimizer = optimizer
        self.tree_params = tree_params or {'max_depth': 3}
        self.scale = bool(scale)
        self.random_state = random_state
        self.poly_degree = poly_degree
        self.auto_tune_poly = auto_tune_poly
        self.max_poly_degree = max_poly_degree

        self.poly = None
        self.scaler_X = StandardScaler() if self.scale else None
        self.scaler_y = StandardScaler() if self.scale else None
        self.tree = None
        self.cof_model = None
        self.n_jobs = n_jobs

    def _ensure_2d(self, arr):
        """Ensure input array is 2D (n_samples, n_features)."""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _preprocess(self, X, fit=True):
        X_proc = self._ensure_2d(X)

        if self.poly is not None:
            if fit:
                X_proc = self.poly.fit_transform(X_proc)
            else:
                X_proc = self.poly.transform(X_proc)

        if self.scaler_X is not None:
            if fit:
                X_proc = self.scaler_X.fit_transform(X_proc)
            else:
                X_proc = self.scaler_X.transform(X_proc)

        return X_proc

    def _evaluate_degree(self, X, y, degree):
        X = self._ensure_2d(X)
        y = self._ensure_2d(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        scaler_X = StandardScaler() if self.scale else None
        scaler_y = StandardScaler() if self.scale else None

        if scaler_X:
            X_train_poly = scaler_X.fit_transform(X_train_poly)
            X_val_poly = scaler_X.transform(X_val_poly)
        if scaler_y:
            y_train = scaler_y.fit_transform(y_train)
            y_val_true = scaler_y.transform(y_val)
        else:
            y_val_true = y_val

        tree = DecisionTreeRegressor(random_state=self.random_state, **self.tree_params)
        tree.fit(X_train_poly, y_train)

        cof_model = COFLeafOptimizer(
            tree=tree,
            optimizer=self.optimizer,
            h_max=2.0,
            random_state=self.random_state,
            scaler_X=scaler_X,
            n_jobs=self.n_jobs
        )
        cof_model.fit(X_train_poly, y_train)
        y_pred_val = cof_model.predict(X_val_poly)

        if scaler_y:
            y_pred_val = scaler_y.inverse_transform(y_pred_val)

        return mean_squared_error(y_val, y_pred_val)

    def fit(self, X_train, y_train):
        X = self._ensure_2d(X_train)
        y = self._ensure_2d(y_train)

        # Auto-tune polynomial degree
        if self.auto_tune_poly:
            scores = {}
            for deg in range(1, self.max_poly_degree + 1):
                try:
                    scores[deg] = self._evaluate_degree(X, y, deg)
                except Exception as e:
                    print(f"⚠️ Skipping degree {deg} due to error: {e}")
                    continue
            if scores:
                self.poly_degree = min(scores, key=scores.get)
                print(f"✅ Selected best polynomial degree: {self.poly_degree}")
            else:
                print("⚠️ No valid polynomial degree found. Using default:", self.poly_degree)

        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)

        X_proc = self._preprocess(X, fit=True)
        y_proc = self.scaler_y.fit_transform(y) if self.scaler_y else y

        self.tree = DecisionTreeRegressor( random_state=self.random_state, **self.tree_params)
        self.tree.fit(X_proc, y_proc)

        self.cof_model = COFLeafOptimizer(
            tree=self.tree,
            optimizer=self.optimizer,
            h_max=2.0,
            random_state=self.random_state,
            scaler_X=self.scaler_X if self.scale else None,
            n_jobs=self.n_jobs
        )
        self.cof_model.fit(X_proc, y_proc)
        return self

    def predict(self, X_test):
        X = self._ensure_2d(X_test)
        X_proc = self._preprocess(X, fit=False)
        y_pred_proc = self.cof_model.predict(X_proc)
        if self.scaler_y:
            return self.scaler_y.inverse_transform(y_pred_proc)
        return y_pred_proc





'''
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from Helping_Code.Optimizers.COFLeafOptimizer import COFLeafOptimizer


class COFPipeline:
    def __init__(self, optimizer="default", max_depth=3, scale=True, random_state=42,
                 poly_degree=1, auto_tune_poly=False, max_poly_degree=3, n_jobs = 2):
        """
        COF Pipeline with optional polynomial expansion and automatic tuning.
        
        Args:
            optimizer (str): Which optimizer to use ("default", "cvxpy", "gurobi").
            max_depth (int): Max depth for the decision tree.
            scale (bool): Whether to standardize X and y.
            random_state (int): Reproducibility.
            poly_degree (int): Default polynomial degree if auto_tune_poly=False.
            auto_tune_poly (bool): If True, searches best poly degree up to max_poly_degree.
            max_poly_degree (int): Maximum polynomial degree to test when auto_tuning.
        """
        self.optimizer = optimizer
        self.max_depth = max_depth
        self.scale = bool(scale)
        self.random_state = random_state
        self.poly_degree = poly_degree
        self.auto_tune_poly = auto_tune_poly
        self.max_poly_degree = max_poly_degree

        self.poly = None
        self.scaler_X = StandardScaler() if self.scale else None
        self.scaler_y = StandardScaler() if self.scale else None
        self.tree = None
        self.cof_model = None
        self.n_jobs = n_jobs

    def _preprocess(self, X, fit=True):
        """Applies PolynomialFeatures + scaling."""
        X_proc = X

        # polynomial expansion
        if self.poly is not None:
            if fit:
                X_proc = self.poly.fit_transform(X_proc)
            else:
                X_proc = self.poly.transform(X_proc)

        # scaling
        if self.scaler_X is not None:
            if fit:
                X_proc = self.scaler_X.fit_transform(X_proc)
            else:
                X_proc = self.scaler_X.transform(X_proc)

        return X_proc

    def _evaluate_degree(self, X, y, degree):
        """Helper: train a quick pipeline with given poly degree and return validation MSE."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                          random_state=self.random_state)

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        scaler_X = StandardScaler() if self.scale else None
        scaler_y = StandardScaler() if self.scale else None

        if scaler_X:
            X_train_poly = scaler_X.fit_transform(X_train_poly)
            X_val_poly = scaler_X.transform(X_val_poly)

        if scaler_y:
            y_train = scaler_y.fit_transform(y_train)
            y_val_true = scaler_y.transform(y_val)
        else:
            y_val_true = y_val

        # Fit a small decision tree
        tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        tree.fit(X_train_poly, y_train)

        # print("Before Leaves Optimization")
        # Leaf optimization
        cof_model = COFLeafOptimizer(tree=tree,
                                     optimizer=self.optimizer,
                                     h_max=2.0,
                                     random_state=self.random_state,
                                     scaler_X=scaler_X,
                                     n_jobs = self.n_jobs)
        cof_model.fit(X_train_poly, y_train)
        # print("After Leaves Optimization")
        y_pred_val = cof_model.predict(X_val_poly)

        if scaler_y:
            y_pred_val = scaler_y.inverse_transform(y_pred_val)

        return mean_squared_error(y_val, y_pred_val)

    def fit(self, X_train, y_train):
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # ---- Auto-tune polynomial degree ----
        if self.auto_tune_poly:
            scores = {}
            for deg in range(1, self.max_poly_degree + 1):
                try:
                    scores[deg] = self._evaluate_degree(X, y, deg)
                except Exception as e:
                    print(f"⚠️ Skipping degree {deg} due to error: {e}")
                    continue
            self.poly_degree = min(scores, key=scores.get)
            print(f"✅ Selected best polynomial degree: {self.poly_degree}")

        # final polynomial object
        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)

        # preprocess
        X_proc = self._preprocess(X, fit=True)
        y_proc = self.scaler_y.fit_transform(y) if self.scaler_y else y

        # decision tree
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        self.tree.fit(X_proc, y_proc)

        # COF optimizer
        self.cof_model = COFLeafOptimizer(
            tree=self.tree,
            optimizer=self.optimizer,
            h_max=2.0,
            random_state=self.random_state,
            scaler_X=self.scaler_X if self.scale else None,
            n_jobs=self.n_jobs
        )
        self.cof_model.fit(X_proc, y_proc)
        return self

    def predict(self, X_test):
        X = np.asarray(X_test)
        X_proc = self._preprocess(X, fit=False)

        y_pred_proc = self.cof_model.predict(X_proc)
        if self.scaler_y:
            return self.scaler_y.inverse_transform(y_pred_proc)
        else:
            return y_pred_proc

'''


'''
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


from Helping_Code.Optimizers.COFLeafOptimizer import COFLeafOptimizer

# ---------- COFPipeline ----------
class COFPipeline:
    def __init__(self, optimizer="default", max_depth=3, scale=True, random_state=42):
        self.optimizer = optimizer
        self.max_depth = max_depth
        self.scale = bool(scale)
        self.random_state = random_state

        self.scaler_X = StandardScaler() if self.scale else None
        self.scaler_y = StandardScaler() if self.scale else None
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        self.cof_model = None

    def fit(self, X_train, y_train):
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.scale:
            X_proc = self.scaler_X.fit_transform(X)
            y_proc = self.scaler_y.fit_transform(y)
        else:
            X_proc, y_proc = X, y

        # Fit tree on processed data
        self.tree.fit(X_proc, y_proc)

        # Create COF optimizer and pass scaler_X for inverse bounds transform
        self.cof_model = COFLeafOptimizer(
            tree=self.tree,
            optimizer=self.optimizer,
            h_max=2.0,
            random_state=self.random_state,
            scaler_X=self.scaler_X if self.scale else None
        )

        # Fit COF on processed data
        self.cof_model.fit(X_proc, y_proc)
        return self

    def predict(self, X_test):
        X = np.asarray(X_test)
        if self.scale:
            X_proc = self.scaler_X.transform(X)
        else:
            X_proc = X

        y_pred_proc = self.cof_model.predict(X_proc)

        if self.scale:
            return self.scaler_y.inverse_transform(y_pred_proc)
        else:
            return y_pred_proc

'''