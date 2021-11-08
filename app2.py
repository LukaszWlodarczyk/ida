from math import sqrt


class Neural:
    def __init__(self):
        self.weights = []

    def set_weights(self, weights):
        self.weights = weights

    def calculate_output(self, input_vector):
        total = 0.0
        for i in range(len(input_vector)):
            total += input_vector[i]*self.weights[i]
        return total


def normalize(vector):
    vector_len = sqrt(sum(vector))
    for i, elem in enumerate(vector):
        vector[i] = elem/vector_len


def predict(vector, neu_x: Neural, neu_y: Neural, neu_z: Neural):
    res_x = neu_x.calculate_output(vector)
    res_y = neu_y.calculate_output(vector)
    res_z = neu_z.calculate_output(vector)
    result = max(res_x,res_y,res_z)
    if result == res_x:
        winner = 'X'
    elif result == res_y:
        winner = 'Y'
    elif result == res_z:
        winner = 'Z'
    else:
        winner = "Cos nie dziala xd"
    return f'Jest to litera {winner} z prawdopodobienstwem {result}.\n' \
        f'X: {res_x}\n' \
        f'Y: {res_y}\n' \
        f'Z: {res_z}'


if __name__ == '__main__':
    x = [1, 0, 0, 1,
         0, 1, 1, 0,
         0, 1, 1, 0,
         1, 0, 0, 1]

    y = [1, 0, 0, 1,
         0, 1, 1, 0,
         0, 1, 0, 0,
         1, 0, 0, 0]

    z = [1, 1, 1, 1,
         0, 0, 1, 0,
         0, 1, 0, 0,
         1, 1, 1, 1]

    normalize(x)
    normalize(y)
    normalize(z)

    neural_x = Neural()
    neural_y = Neural()
    neural_z = Neural()

    neural_x.set_weights(x)
    neural_y.set_weights(y)
    neural_z.set_weights(z)

    new_letter = [1, 0, 0, 1,
                  0, 1, 1, 0,
                  0, 1, 1, 0,
                  1, 0, 0, 1]
    normalize(new_letter)

    print(predict(new_letter, neural_x, neural_y, neural_z))
