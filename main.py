# import torch
# import matplotlib.pyplot as plt

from domains import Rectangle, test_rectangle, test_sphere
from benchmark import Linear, SteamGovernor , read_benchmark, generate_benchmark_dataset

import numpy as np
import pandas as pd
import torch


from helping_function.plot.CustomPlotClass import CustomPlotClass

if __name__ == "__main__":


    
    # test_rectangle()
    # test_sphere()
    # benchmark = read_benchmark("steam")
    # steamGovernor = SteamGovernor()
    # steamGovernor.domain = Rectangle([-3, -3, -3], [3, 3, 3])
    # print("Benchmark:", steamGovernor.name, "Dimension:", steamGovernor.dimension)

    # data =  steamGovernor.get_data(15)
    # print("Sample data:\n", data.numpy())

    # image = benchmark.get_image()
    # print("Image at midpoint:", image)

    # output = steamGovernor.f(data[0].tolist())
    # print("The output of generated data:\n", output)

    # benchmark_name = "lin"
    # X, Y = generate_benchmark_dataset(benchmark_name, n_samples=100000)
    # print(f"Benchmark: {benchmark_name}")
    # print("Sample inputs:\n", X.numpy())
    # print("Sample outputs:\n", Y.numpy())

    benchmark = SteamGovernor()
    #benchmark.domain = Rectangle(lb=[-300,-300, -300], ub=[300,300,300])
    X = benchmark.get_data(10000000)
    # Compute outputs
    Y_list = [benchmark.f(x.tolist()) for x in X]
    Y = torch.tensor(Y_list, dtype=torch.float32)
    print("Data is generated")

    # Combine horizontally
    data = np.hstack((X, Y))

    # Create column names
    x_cols = [f"X{i+1}" for i in range(X.shape[1])]
    y_cols = [f"Y{i+1}" for i in range(Y.shape[1])]
    columns = x_cols + y_cols

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("Dataset/SteamGovernor/data_SteamGovernor_10000000.csv" , index=False)

    print("Data is saved")