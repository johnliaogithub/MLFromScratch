import numpy as np

class LinearRegression: 

    def __init__(self, input_dimensions: int): 
        assert(input_dimensions >= 0)

        # initialize state
        self.M = np.random.randn(input_dimensions)
        self.b = np.random.randn()

    def eval(self, input: np.array): 
        return self.M * input + self.b


if __name__ == "__main__": 
    # test
    x = np.array([1, 2, 3, 4, 5])

    model = LinearRegression(5)
    print("eval: ", model.eval(x))