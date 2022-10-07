import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv("world-happiness-report-2017.csv")
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)
input_para_name ='Economy..GDP.per.Capita.'
output_para_name ='Happiness.Score'
x_train = train_data[[input_para_name]].values
y_train = train_data[[output_para_name]].values
x_test = test_data[input_para_name].values
y_test = test_data[output_para_name].values







num_iterations =500
learning_rate = 0.01
linear_regressor = LinearRegression(x_train, y_train)
linear_regressor.train(learning_rate,num_iterations)
(theta, cost_history) = linear_regressor.train(learning_rate, num_iterations)
print("cost at the beginning", cost_history[0])
print("cost after the training", cost_history[-1])
print(theta)

plt.plot(range(num_iterations),cost_history)
plt.xlabel("Iter")
plt.ylabel("cost_history")
plt.show()

prediction_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), prediction_num).reshape(prediction_num,1)
y_predictions = linear_regressor.predict(x_predictions)
plt.scatter(x_train, y_train, label = 'Train data')
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_predictions, y_predictions, 'r', label='predicted')
plt.xlabel(input_para_name)
plt.ylabel(output_para_name)
plt.title('happy')
plt.legend()
plt.show()

