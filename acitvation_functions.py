import math
def linear_activation(x):
    return x

def step_activation(x):
    return 1 if x >= 0 else 0

def sigmoid_activation(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def relu_activation(x):
    return max(0, x)

def leaky_relu_activation(x):
    return max(0.01 * x, x)

def  elu_activation(x, alpha = 1):
    return alpha * (math.exp(x) - 1) if x < 0 else x

def tanh_activation(x):
    return math.tanh(x)

def softmax(scores):
    max_output = max(scores)

    exponentials = [math.exp(score - max_output) for score in scores]

    total = sum(exponentials)

    return [value/total for value in exponentials]


def tanh_derivative(x):
    return 1 - tanh_activation(x)**2

def sigmoid_derivative(x):
    s = sigmoid_activation(x)
    return s * (1 - s)

def relu_derivative(x):
    return 1 if x > 0 else 0

def elu_derivative(x, alpha=1):
    return alpha * math.exp(x) if x < 0 else 1

def linear_derivation(x):
    return 1

def leaky_relu_derivative(x):
    return 1 if x > 0 else 0.01