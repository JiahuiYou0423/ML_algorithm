import numpy as np
import pandas as pd
import plotly

import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv("world-happiness-report-2017.csv")
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)
input_para_name1 ='Economy..GDP.per.Capita.'
input_para_name2 ='Freedom'
output_para_name ='Happiness.Score'
x_train = train_data[[input_para_name1,input_para_name2]].values
y_train = train_data[[output_para_name]].values
x_test = test_data[[input_para_name1, input_para_name2]].values
y_test = test_data[[output_para_name]].values


import plotly.graph_objects as go
# Configure the plot with training dataset.
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)


plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)


plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_para_name1},
        'yaxis': {'title': input_para_name2},
        'zaxis': {'title': output_para_name}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]

plot_figure = go.Figure(data=plot_data, layout=plot_layout)

plotly.offline.iplot(plot_figure)
num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0

linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

print('start to loss',cost_history[0])
print('final loss',cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

predictions_num = 10

x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()

y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()


x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)


x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2,
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.iplot(plot_figure)



