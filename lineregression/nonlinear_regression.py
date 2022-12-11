import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .lr import LinearRegression

data = pd.read_csv("csv")

x = data['x'].values.reshape((data.shape[0], 1))
y = data['y'].values.reshape((data.shape[0], 1))

data.head(10)

plt.plot(x, y)
plt.show()

num_iterations = 50000
learning_rate = 0.02
polynomial_degree = 15
sinusoid_degree = 15
normalize_data = True

linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)
