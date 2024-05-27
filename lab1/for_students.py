import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
X = np.column_stack((np.ones((len(x_train), 1)), x_train))

# dot - oblicza iloczyn skalarny
# linalg.inv - oblicza macierz odwrotna
# .T - macierz transponowana
# theta = (X^T * X)^-1 * X^T * y
theta_best = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y_train)

# TODO: calculate error
mse = 0
for i in range(np.size(y_test)):
    mse += (theta_best[0] + theta_best[1] * x_test[i] - y_test[i]) ** 2
mse = mse / np.size(y_test)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# srednia wartosc i odchylenie standardowe

x_train_normalized = (x_train - np.mean(x_train)) / np.std(x_train)
x_test_normalized = (x_test - np.mean(x_train)) / np.std(x_train)

y_train_normalized = (y_train - np.mean(y_train)) / np.std(y_train)
y_test_normalized = (y_test - np.mean(y_train)) / np.std(y_train)

# TODO: calculate theta using Batch Gradient Descent
theta = np.random.rand(2)
learning_rate = 0.1

mse_gd = 0
for i in range(np.size(y_test)):
    mse_gd += (theta[0] + theta[1] * x_test_normalized[i] - y_test_normalized[i]) ** 2
mse_gd = mse_gd / np.size(y_test)
prev_mse_gd = mse_gd

while prev_mse_gd >= mse_gd:

    X_normalized = np.column_stack((np.ones((len(x_train_normalized), 1)), x_train_normalized))

    # gradient = 2/m(X^T(X.Theta-y))
    gradients = 2 / np.size(y_train_normalized) * X_normalized.T.dot(
        X_normalized.dot(theta) - y_train_normalized)

    theta = theta - learning_rate * gradients
    prev_mse_gd = mse_gd

    mse_gd = 0
    for i in range(np.size(y_test)):
        mse_gd += (theta[0] + theta[1] * x_test_normalized[i] - y_test_normalized[i]) ** 2
    mse_gd = mse_gd / np.size(y_test)

# TODO: calculate error
mse_gradient = 0
for i in range(np.size(x_test_normalized)):
    mse_gradient += (theta[0] + theta[1] * x_test_normalized[i] - y_test_normalized[i]) ** 2
mse_gradient = mse_gradient / np.size(x_test_normalized)

# plot the regression line
x = np.linspace(min(x_test_normalized), max(x_test_normalized), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test_normalized, y_test_normalized)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

print("Theta first:", theta_best)
print("Theta second:", theta)
print("MSE: ", mse)
print("MSE gradient: ", mse_gradient)
