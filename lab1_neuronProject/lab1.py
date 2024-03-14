from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(int(datetime.now().timestamp()))


def initialize_inputs_weights_biases(input_size, layer_size, biases):
    inputs = np.random.uniform(-3, 3, size=(15, input_size))
    weights = np.random.uniform(-1, 1, size=(layer_size, input_size))
    coef = biases
    return inputs, weights, coef


def function_neuron(inputs, weights, coef):
    s = np.dot(weights, inputs)
    s = np.sum(s) + coef
    return np.where(s > 0, 1, 0)


def plot_perceptron(inputs, weights, coef):
    for input_point in inputs:
        output = function_neuron(input_point, weights, coef)
        color = 'blue' if output > 0 else 'red'
        plt.plot(input_point[0], input_point[1],
                 marker='o', markersize=10, color=color)

    column_sum_1 = np.sum(weights[:, 0])
    column_sum_2 = np.sum(weights[:, 1])

    # Обчислення параметрів лінії розділення
    slope = -column_sum_1 / column_sum_2
    intercept = -coef / column_sum_2

    x = np.linspace(-3, 3, 100)
    y = slope * x + intercept

    plt.plot(x, y)
    plt.xlabel('Вхід 1')
    plt.ylabel('Вхід 2')
    plt.title('Класифікація векторів входу')
    plt.show()
    print(f"Функція: {column_sum_1}p1 + {column_sum_2}p2 + {coef}")


input_size = 2
layer_size = 2
biases = 0

inputs, weights, coef = initialize_inputs_weights_biases(input_size, layer_size, biases)
plot_perceptron(inputs, weights, coef)

biases = 2
inputs, weights, coef = initialize_inputs_weights_biases(input_size, layer_size, biases)
plot_perceptron(inputs, weights, coef)
