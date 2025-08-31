from typing import Callable, List, Union
import numpy as np
import torch

import pandas as pd
import os

from domains import Rectangle, Sphere

# ----------------- Benchmark Base Class -----------------

class Benchmark:
    def __init__(self) -> None:
        self.dimension: int = None
        self.name: str = None
        self.domain: Union[Rectangle, Sphere] = None
        self.scale: List[float] = None
        self.image: List[float] = None

    def f(self, v: List[float]) -> List[float]:
        raise NotImplementedError

    def get_domain(self, x: List[float], _And: Callable):
        return self.domain.generate_domain(x, _And)

    def get_data(self, n=10000):
        return self.domain.generate_bloated_data(n)

    def get_image(self):
        """Find the image of the function using midpoints"""
        if isinstance(self.domain, Sphere):
            raise ValueError("Intervals not implemented for spherical domains")
        return self.f(self.domain.as_intervals())

    def f_intervals(self, v: List[float]):
        return self.f(v)


# ----------------- Benchmarks -----------------

class Linear(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Linear"
        self.short_name = "lin"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1, 1]
        self.image = self.get_image()

    def f(self, v):
        x, y = v
        return [-x + y, -y]


class SteamGovernor(Benchmark):
    def __init__(self) -> None:
        self.dimension = 3
        self.name = "Steam"
        self.short_name = "steam"
        self.domain = Rectangle([-1, -1, -1], [1, 1, 1])
        self.scale = [1, 1, 1]
        self.image = self.get_image()

    def f(self, v):
        x, y, z = v
        f = [
            y,
            z**2 * np.sin(x) * np.cos(x) - np.sin(x) - 3 * y,
            -(np.cos(x) - 1),
        ]
        return [fi / si for fi, si in zip(f, self.scale)]


class JetEngine(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Jet Engine"
        self.short_name = "jet"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1, 1]
        self.image = self.get_image()

    def f(self, v):
        x, y = v
        return [-y - 1.5 * x**2 - 0.5 * x**3 - 0.1, 3 * x - y]


class NL1(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Non-Lipschitz1"
        self.short_name = "nl1"
        self.domain = Rectangle([0, -1], [1, 1])
        self.scale = [1, 1]

    def f(self, v):
        x, y = v
        return [y, np.sqrt(max(x, 0.0))]


class NL2(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Non-Lipschitz2"
        self.short_name = "nl2"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1, 1]

    def f(self, v):
        x, y = v
        return [x**2 + y, np.cbrt(x**2) - x]
        # if isinstance(x, (np.ndarray, float)):   # numpy or float input
        #     return [x**2 + y, np.cbrt(x**2) - x]
        # elif isinstance(x, torch.Tensor):        # torch input
        #     return [x**2 + y, torch.pow(x**2, 1/3) - x]
        # else:
        #     raise TypeError(f"Unsupported type {type(x)} for NL2.f")

class WaterTank(Benchmark):
    def __init__(self) -> None:
        self.dimension = 1
        self.name = "Water-tank"
        self.short_name = "tank"
        self.domain = Rectangle([0.1], [10])
        self.scale = [1]

    def f(self, v):
        x = v[0] if isinstance(v, (list, tuple)) else v
        return [1.5 - np.sqrt(x)]


class Exponential(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Exponential"
        self.short_name = "exp"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1, 1]
        self.image = self.get_image()

    def f(self, v):
        x, y = v
        return [-np.sin(np.exp(y**3 + 1)) - y**2, -x]


class VanDerPol(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Van der Pol"
        self.short_name = "vdp"
        self.domain = Rectangle([-3, -3], [3, 3])
        self.scale = [1, 1]
        self.mu = 1.0
        self.image = self.get_image()

    def f(self, v):
        x, y = v
        return [-y, self.mu * (1 - x**2) * y - x]


class Sine2D(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Sine2D"
        self.short_name = "sine2d"
        self.domain = Rectangle([-2, -2], [2, 2])
        self.scale = [1, 1]
        self.freq_y = 1.0
        self.freq_x = 1.0
        self.image = self.get_image()

    def f(self, v):
        x, y = v
        return [np.sin(self.freq_y * y), -np.sin(self.freq_x * x)]


class NonlinearOscillator(Benchmark):
    def __init__(self) -> None:
        self.dimension = 1
        self.name = "Nonlinear Oscillator"
        self.short_name = "nonlin-osc"
        self.domain = Rectangle([-3.0], [3.0])
        self.scale = [1]
        self.linear_coeff = 1.0
        self.cubic_coeff = 0.5
        self.sine_coeff = 0.3

    def f(self, v):
        x = v[0] if isinstance(v, (list, tuple)) else v
        return [-self.linear_coeff * x - self.cubic_coeff * (x**3) + self.sine_coeff * np.sin(x)]


# ----------------- Helper Function -----------------

def read_benchmark(name: str):
    mapping = {
        "lin": Linear,
        "exp": Exponential,
        "steam": SteamGovernor,
        "jet": JetEngine,
        "nl1": NL1,
        "nl2": NL2,
        "tank": WaterTank,
        "vdp": VanDerPol,
        "sine2d": Sine2D,
        "nonlin-osc": NonlinearOscillator,
    }
    return mapping[name]() if name in mapping else None



def generate_benchmark_dataset(benchmark_name: str, n_samples: int = 10000):
    """
    Loads a benchmark and generates input-output data.

    Args:
        benchmark_name (str): Name of the benchmark ('lin', 'exp', 'steam', etc.)
        n_samples (int): Number of input points to generate

    Returns:
        X (torch.Tensor): Input points of shape (n_samples, dimension)
        Y (torch.Tensor): Output points of shape (n_samples, dimension)
    """
    benchmark = read_benchmark(benchmark_name)
    if benchmark is None:
        raise ValueError(f"Benchmark '{benchmark_name}' not found!")

    
    # Generate input data
    X = benchmark.get_data(n_samples)

    # Compute outputs
    Y_list = [benchmark.f(x.tolist()) for x in X]
    Y = torch.tensor(Y_list, dtype=torch.float32)

    return X, Y

def generate_save_benchmark_dataset(benchmark_name: str, n_samples: int = 10000):
    """
    Loads or generates benchmark dataset and saves/loads it from disk.

    Args:
        benchmark_name (str): Name of the benchmark ('lin', 'exp', 'steam', etc.)
        n_samples (int): Number of input points to generate

    Returns:
        X (torch.Tensor): Input points of shape (n_samples, dimension)
        Y (torch.Tensor): Output points of shape (n_samples, dimension)
    """
    # -----------------
    # Paths
    folder_path = f"Dataset/{benchmark_name}/{benchmark_name}_{n_samples}"
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"data_{benchmark_name}_{n_samples}.csv")

    # -----------------
    # Case 1: Load if exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        X_cols = [col for col in df.columns if col.startswith("X")]
        Y_cols = [col for col in df.columns if col.startswith("Y")]

        # X = torch.tensor(df[X_cols].values, dtype=torch.float32)
        # Y = torch.tensor(df[Y_cols].values, dtype=torch.float32)
        X = df[X_cols].to_numpy(dtype=np.float32)
        Y = df[Y_cols].to_numpy(dtype=np.float32)
        print(f"Loaded dataset from {file_path}")
        return X, Y

    # -----------------
    # Case 2: Generate if not exists
    benchmark = read_benchmark(benchmark_name)
    if benchmark is None:
        raise ValueError(f"Benchmark '{benchmark_name}' not found!")

    # Generate input data
    X = benchmark.get_data(n_samples)
    X = X.numpy()

    # Compute outputs
    Y_list = [benchmark.f(x.tolist()) for x in X]
    Y = np.array(Y_list)
    # torch.tensor(Y_list, dtype=torch.float32)

    # Combine horizontally
    data = np.hstack((X, Y))

    # Create column names
    x_cols = [f"X{i+1}" for i in range(X.shape[1])]
    y_cols = [f"Y{i+1}" for i in range(Y.shape[1])]
    columns = x_cols + y_cols

    # Save DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)
    print(f"Generated and saved dataset to {file_path}")

    return X, Y