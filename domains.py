from typing import Callable, List, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from domain_plot import plot_rectangle, plot_sphere

# ----------------- Domain Classes -----------------

def square_init_data(domain, batch_size):
    """Samples points uniformly over a rectangular domain."""
    r1 = torch.tensor(domain[0], dtype=torch.float32)
    r2 = torch.tensor(domain[1], dtype=torch.float32)
    return (r1 - r2) * torch.rand(batch_size, len(domain[0])) + r2


def n_dim_sphere_init_data(centre, radius, batch_size):
    """Generates points in a n-dim sphere: ||x - centre|| <= radius"""
    dim = len(centre)
    u = torch.randn(batch_size, dim)
    norm = torch.sqrt(torch.sum(u ** 2, dim=1))
    r = radius * torch.rand(batch_size, dim) ** (1.0 / dim)
    x = (r * u) / norm[:, None] + torch.tensor(centre)
    return x


class Rectangle:
    def __init__(self, lb: List[float], ub: List[float]):
        self.name = "rectangle"
        self.lower_bounds = lb
        self.upper_bounds = ub
        self.dimension = len(lb)

    def generate_domain(self, x: List, _And):
        lower = _And(*[self.lower_bounds[i] <= x[i] for i in range(self.dimension)])
        upper = _And(*[x[i] <= self.upper_bounds[i] for i in range(self.dimension)])
        return _And(lower, upper)

    def generate_data(self, batch_size: int) -> torch.Tensor:
        return square_init_data([self.lower_bounds, self.upper_bounds], batch_size)

    def generate_bloated_data(self, batch_size: int, bloat=0.1) -> torch.Tensor:
        return square_init_data(
            [
                [lb - bloat for lb in self.lower_bounds],
                [ub + bloat for ub in self.upper_bounds],
            ],
            batch_size=batch_size,
        )

    def as_intervals(self):
        """Return midpoints for numeric evaluation"""
        return [(lb + ub) / 2 for lb, ub in zip(self.lower_bounds, self.upper_bounds)]

    def check_interior(self, S: torch.Tensor) -> torch.Tensor:
        lb = torch.tensor(self.lower_bounds)
        ub = torch.tensor(self.upper_bounds)
        return torch.logical_and(S > lb, S < ub).all(dim=1)

    def scale(self, factor: List):
        lb = [bound * s for bound, s in zip(self.lower_bounds, factor)]
        ub = [bound * s for bound, s in zip(self.upper_bounds, factor)]
        return Rectangle(lb, ub)

    def shift(self, shift):
        lb = [bound + s for bound, s in zip(self.lower_bounds, shift)]
        ub = [bound + s for bound, s in zip(self.upper_bounds, shift)]
        return Rectangle(lb, ub)


class Sphere:
    def __init__(self, centre: List[float], radius: float):
        self.name = "sphere"
        self.centre = centre
        self.radius = radius
        self.dimension = len(centre)

    def generate_domain(self, x: List, _And):
        return _And(sum([(x[i] - self.centre[i]) ** 2 for i in range(self.dimension)]) <= self.radius ** 2)

    def generate_data(self, batch_size):
        return n_dim_sphere_init_data(self.centre, self.radius ** 2, batch_size)

    def as_intervals(self):
        """Return center for numeric evaluation"""
        return self.centre






# ----------------- Helper Function -----------------

def test_rectangle():
    print("=== Testing Rectangle ===")
    rect = Rectangle(lb=[-1, -2, -3], ub=[1, 2, 3])  # 2D rectangle
    
    # Generate 5 random points
    data = rect.generate_data(batch_size=5)
    print("Random samples in rectangle:\n", data.numpy())

    # Generate bloated data
    # data_bloated = rect.generate_bloated_data(batch_size=5, bloat=0.5)
    # print("Bloated samples (with margin):\n", data_bloated)

    # Check if points are inside
    # inside = rect.check_interior(data)
    # print("Check if inside rectangle:\n", inside)

    # Scale and shift
    # rect_scaled = rect.scale([2, 0.5])
    # rect_shifted = rect.shift([1, 1])
    # print("Scaled Rectangle bounds:", rect_scaled.lower_bounds, rect_scaled.upper_bounds)
    # print("Shifted Rectangle bounds:", rect_shifted.lower_bounds, rect_shifted.upper_bounds)

    # Midpoints / intervals
    print("Rectangle as intervals (midpoints):", rect.as_intervals())

    ax = plot_rectangle(rect, color="green", alpha=0.5)
    plt.title("Interactive 3D Rectangle")
    plt.show(block=True)
    # plt.show()


def test_sphere():
    print("\n=== Testing Sphere ===")
    sphere = Sphere(centre=[0, 0, 0], radius=1.0)  # 3D unit sphere
    
    # Generate 5 random points
    data = sphere.generate_data(batch_size=5)
    print("Random samples in sphere:\n", data)

    # Domain check formula (dummy, using Python's and instead of dReal)
    x = torch.tensor([0.1, 0.2, 0.3])  # a point
    in_sphere = sum([(x[i] - sphere.centre[i])**2 for i in range(sphere.dimension)]) <= sphere.radius**2
    print("Point", x.tolist(), "inside sphere?", in_sphere.item())

    # Center
    print("Sphere as intervals (center):", sphere.as_intervals())

    
    ax = plot_sphere(sphere, color="purple", alpha=0.5)
    plt.title("Interactive 3D Sphere")
    plt.show(block=True)
