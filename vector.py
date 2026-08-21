class Vector:
    def __init__(self, data):
        self.data = data
        self.size = len(data)
    def add(self, other):
        result = [self.data[i] + other.data[i] for i in range(self.size)]
        return Vector(result)
    def subtract(self, other):
        result = [self.data[i] - other.data[i] for i in range(self.size)]
        return Vector(result)
    def multiply(self, scalar):
        result = [value * scalar for value in self.data]
        return Vector(result)
    def dot(self, other):
        result = sum(self.data[i] * other.data[i] for i in range(self.size))
        return result
    def element_wise_multiplication(self, other):
        result = [self.data[i] * other.data[i] for i in range(self.size)]
        return Vector(result)
