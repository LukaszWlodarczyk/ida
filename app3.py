from math import e, pow
from random import uniform


def sigmoid(x):
    x = x * (-1.0)
    tmp = 1 + pow(e, x)
    return 1 / tmp


def d_sigmoid(x):
    x = x * (-1.0)
    tmp = pow(e,x)
    tmp2 = pow((1+tmp),2)
    return tmp/tmp2


class Neural:
    def __init__(self, activation, d_activation, learning_step, name):
        self.position = name
        self.weights = []
        self.inputs = []
        self.output = 0.0
        self.activation = activation
        self.d_activation = d_activation
        self.excepted_output = None
        self.learning_step = learning_step
        self.bias = 1.0
        self.output_before_activation = 0.0

    def __str__(self):
        return f"Neuron {self.position}: \n" \
            f"Input: {self.inputs} + {self.bias} \n" \
            f"Weights: {self.weights} \n" \
            f"Output: {self.output}"

    def set_weights(self, new_weights: list):
        self.weights = []
        for weight in new_weights:
            self.weights.append(weight)

    def calculate(self, inputs):
        self.inputs = inputs
        self.output = 0.0
        if len(self.weights) == 0:
            self.init_weights(inputs)
        for w, x in zip(self.weights, inputs):
            self.output += w * x
        self.output = self.output + self.weights[-1]*self.bias
        self.output_before_activation = self.output
        self.output = self.activation(self.output)
        return self.output

    def init_weights(self, inputs):
        for w in range(len(inputs)+int(self.bias)):
            self.weights.append(uniform(0, 1))

    def calculate_error(self):
        error = -1*(self.excepted_output - self.output)
        return error

    def calculate_error_signal_in_output_layer(self):
        error_signal = self.output*(1.0-self.output)*self.calculate_error()
        return error_signal

    def update_weights_in_output_layer(self):
        for i, weight in enumerate(self.inputs):
            self.weights[i] = self.weights[i] - self.learning_step*self.calculate_error_signal_in_output_layer()\
                              * self.inputs[i]
        if self.bias == 1:
            self.weights[-1] = self.weights[-1] - self.learning_step*\
                               self.calculate_error_signal_in_output_layer()

    def update_weights_in_hidden_layer(self, next_layer, error_on_each_output):
        tmp = 0.0
        for i, dataset in enumerate(error_on_each_output):
            tmp += dataset[0]*dataset[1]*next_layer.neurons[i].weights[self.position]
        after_first_step = tmp
        after_first_step *= self.output * (1.0 - self.output)
        self.weights[-1] = self.weights[-1] - self.learning_step * after_first_step
        after_first_step = tmp
        for i in range(len(self.weights)-1):
            after_first_step *= self.output*(1.0-self.output)*self.inputs[i]
            self.weights[i] = self.weights[i] - self.learning_step*after_first_step
            after_first_step = tmp


class Layer:
    def __init__(self, activation, d_activation, learning_step=0.8, next_layer=None):
        self.neurons = []
        self.activation = activation
        self.d_activation = d_activation
        self.non_activation_output = []
        self.output = []
        self.excepted_output = []
        self.learning_step = learning_step
        self.next_layer = next_layer
        self.output_before_activation = []

    def __str__(self):
        return f"{self.neurons}"

    def add_neurons(self, quantity):
        for i in range(quantity):
            neural = Neural(self.activation, self.d_activation, self.learning_step, i)
            self.neurons.append(neural)

    def calculate_error_and_d_activate_on_each_output(self):
        error_and_d_activate_on_each_output = []
        for i in range(len(self.excepted_output)):
            error = -1 * (self.excepted_output[i] - self.output[i])
            d_activation_value = self.d_activation(self.output_before_activation[i])
            error_and_d_activate_on_each_output.append((error, d_activation_value))
        return error_and_d_activate_on_each_output

    def output_error(self):
        tmp = 0.0
        for i in range(len(self.excepted_output)):
            tmp += abs(self.excepted_output[i]-self.output[i])
        return tmp

    def run_epoch(self, inputs):
        self.output = []
        self.output_before_activation = []
        if self.next_layer is None:
            for neural in self.neurons:
                result = neural.calculate(inputs)
                self.output.append(result)
                self.output_before_activation.append(neural.output_before_activation)
        else:
            for neural in self.neurons:
                result = neural.calculate(inputs)
                self.output.append(result)

    def set_excepted_output(self, excepted_output):
        self.excepted_output = excepted_output
        for i, neural in enumerate(self.neurons):
            neural.excepted_output = self.excepted_output[i]

    def update(self, error_and_d_activate_on_each_output):
        if self.next_layer is None:
            for neural in self.neurons:
                neural.update_weights_in_output_layer()
        else:
            for neural in self.neurons:
                neural.update_weights_in_hidden_layer(self.next_layer, error_and_d_activate_on_each_output)


class Network:
    def __init__(self, learning_step = 0.5):
        self.layers = []
        self.error_and_d_activate_on_each_output = []
        self.learning_step = learning_step

    def add_layer(self, layer):
        self.layers.append(layer)

    def calculate_error_and_d_activate_on_each_output(self, error_on_each_output):
        self.error_and_d_activate_on_each_output = error_on_each_output


inputs = [
    ([1, 0, 0, 0], [1, 0, 0, 0]),
    ([0, 1, 0, 0], [0, 1, 0, 0]),
    ([0, 0, 1, 0], [0, 0, 1, 0]),
    ([0, 0, 0, 1], [0, 0, 0, 1])
]


network = Network(1000)

# inputs = [([0.05, 0.1], [0.01, 0.99])]
output = Layer(sigmoid, d_sigmoid)
hidden = Layer(sigmoid, d_sigmoid, next_layer=output)

hidden.add_neurons(2)
output.add_neurons(4)

# hidden.neurons[0].set_weights([0.15, 0.2, 0.35])
# hidden.neurons[1].set_weights([0.25, 0.3, 0.35])
# output.neurons[0].set_weights([0.4, 0.45, 0.60])
# output.neurons[1].set_weights([0.5, 0.55, 0.60])

network.add_layer(hidden)
network.add_layer(output)
for epoch in range(10001):
    for dataset in inputs:
        hidden.run_epoch(dataset[0])
        output.set_excepted_output(dataset[1])
        output.run_epoch(hidden.output)
        network.calculate_error_and_d_activate_on_each_output(output.calculate_error_and_d_activate_on_each_output())
        output.update(network.error_and_d_activate_on_each_output)
        hidden.update(network.error_and_d_activate_on_each_output)
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch}: \n"
                  f"Error: {output.output_error()}\n"
                  f"Current output: {output.output}\n"
                  f"Excepted output: {output.excepted_output}")



# print(hidden.neurons[0])
# print(hidden.neurons[1])
# print(output.neurons[0])
# print(output.neurons[1])
print(output.output)