import math
class relu:
    def __call__(self, x):
        return max(0, x)
    def derivative(self, x):
        return 1 if x > 0 else 0
class leaky_relu:
    def __call__(self, x):
        return max(0.01 * x, x)
    def derivative(self, x):
        return 1 if x > 0 else 0.01
class elu:
    def __init__(self, alpha=1):
        self.alpha = alpha
    def __call__(self, x):
        return self.alpha * (math.exp(x) - 1) if x < 0 else x
    def derivative(self, x):
        return self.alpha * math.exp(x) if x < 0 else 1
class tanh:
    def __call__(self, x):
        return math.tanh(x)
    def derivative(self, x):
        return 1 - self.__call__(x)**2
class linear:
    def __call__(self, x):
        return x
    def derivative(self, x):
        return 1
class softmax:
    def __call__(self, scores):
        max_output = max(scores)
        exponentials = [math.exp(score - max_output) for score in scores]
        total = sum(exponentials)
        return [value/total for value in exponentials]
class step:
    def __call__(self, x):
        return 1 if x >= 0 else 0
    def derivative(self, x):
        return 0
