from random import uniform


class Neural:
    def __init__(self, inputs):
        self.inputs = inputs
        self.values = []
        self.weights = []
        self.excepted_output = 0.0
        self.output = 0.0
        self.step = 0.2

    def put_data(self, data, excepted_output):
        for value in data:
            self.values.append(value)
        self.excepted_output = excepted_output

    def update_data(self, data, excepted_output):
        for i, value in enumerate(data):
            self.values[i] = value
        self.excepted_output = excepted_output

    def init_weights(self, min_weight, max_weight):
        for w in range(len(self.values)):
            self.weights.append(uniform(min_weight, max_weight))

    def calculate_output(self):
        self.output = 0.0
        for w, x in zip(self.weights, self.values):
            self.output += w * x

    def update_weights(self):
        for i, weight in enumerate(self.weights):
            self.weights[i] = self.weights[i] + self.step * (self.excepted_output - self.output) * self.values[i]

    def __str__(self):
        return f"Weights: {self.weights}, Output: {self.output}, Excepted output: {self.excepted_output}"


def single_pattern(inputs, epochs, step, min_weight, max_weight, min_input, max_input):
    neural = Neural(inputs)
    neural.step = step
    input_values = [uniform(min_input, max_input) for x in range(inputs)]
    excepted_output = uniform(min_input, max_input)
    neural.put_data(input_values, excepted_output)
    neural.init_weights(min_weight, max_weight)
    for i in range(epochs):
        neural.calculate_output()
        neural.update_weights()
        print(f"{i }: {neural}")


def multi_pattern(inputs, epochs, step, min_weight, max_weight, min_input, max_input,
                  custom_input=None, custom_excepted_results=None):
    neural = Neural(inputs)
    neural.step = step
    input_patterns = []
    excepted_outputs = []
    if custom_input is not None:
        input_patterns = custom_input
        excepted_outputs = custom_excepted_results
    else:
        for i in range(epochs):
            input_patterns.append([uniform(min_input, max_input) for x in range(inputs)])
            excepted_outputs.append(uniform(min_input, max_input))

    neural.put_data(input_patterns[0], excepted_outputs[0])
    neural.init_weights(min_weight, max_weight)
    neural.calculate_output()
    neural.update_weights()
    for x in range(epochs):
        for i in range(len(input_patterns)):
            neural.update_data(input_patterns[i], excepted_outputs[i])
            neural.calculate_output()
            neural.update_weights()
            print(f"Epoch: {x}, Pattern: {i}, Inputs: {neural.values}, {neural}")


def pattern_loader(filename):
    skipper = 2
    input_patterns = []
    excepted_results = []
    with open(filename) as f:
        for line in f:
            if skipper > 0:
                skipper -= 1
            else:
                line = line.replace("-"," -")
                *x,y = map(float,line.split("   "))
                input_patterns.append(x)
                excepted_results.append(y)
    return input_patterns, excepted_results


if __name__ == "__main__":
    input_patterns, excepted_results = pattern_loader("examples/patterns6.txt")
    INPUTS = len(input_patterns[0])
    EPOCHS = 1000
    STEP = 0.001
    MIN_WEIGHT = 0.0
    MAX_WEIGHT = 1.0
    MIN_INPUT = -1.0
    MAX_INPUT = 1.0

    # # single_pattern(INPUTS,EPOCHS,STEP,MIN_WEIGHT,MAX_WEIGHT,MIN_INPUT,MAX_INPUT)
    multi_pattern(INPUTS,EPOCHS,STEP,MIN_WEIGHT,MAX_WEIGHT,MIN_INPUT,MAX_INPUT, input_patterns, excepted_results)

