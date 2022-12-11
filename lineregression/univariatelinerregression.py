import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .lr import LinearRegression

data = pd.read_csv("csv")

# 训练集和测试集
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

input_param_name = "feature column"
output_param_name = "label column"

x_train = train_data[[input_param_name]].values # [n; 1]
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values # [n; 1]
y_test = test_data[[output_param_name]].values

plt.scatter(x_train, y_train, label="Train data")
plt.scatter(x_test, y_test, label='test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("feature")
plt.legend()
plt.show()

num_iteration=500
learning_rate = 0.01
leaner_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = leaner_regression.train(learning_rate, num_iteration)


plt.plot(range(num_iteration), cost_history)
plt.xlabel('iter')
plt.ylabel('loss(cost)')
plt.title('GD')
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num)
y_predictions = leaner_regression.predict(x_predictions)
plt.scatter(x_train, y_train, label="Train data").reshape(predictions_num, 1)
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("feature")
plt.legend()
plt.show()