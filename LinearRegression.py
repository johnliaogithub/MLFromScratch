import numpy as np

class LinearRegression: 

    def __init__(self, input_dimensions: int):      # x
        assert(input_dimensions >= 0)

        self.input_dimensions = input_dimensions

        # initialize state
        self.M = np.random.randn(input_dimensions).reshape((input_dimensions, 1))  # (x, 1)
        self.b = np.random.randn()                  # (1, )

    def eval(self, features: np.array, targets: np.array):
        return np.average(np.abs(self.feedforward(features) - targets))

    def feedforward(self, features: np.array): 
        return features @ self.M + self.b           # (n, 1)

    def train(self, features: np.array, targets: np.array, epochs:int = 0, learning_rate:float = 0.01): 
        assert(len(features.shape) == len(targets.shape) == 2)
        assert(features.shape == (targets.size, self.input_dimensions))

        n = features.shape[0]

        for i in range(epochs): 
            feedfoward = self.feedforward(features)

            mse = np.average(np.square(targets - feedfoward))

            print("Training loop: {}, MSE: {}".format(i, mse))

            # gradient of feedforward
            g_f = 2 / n * (feedfoward - targets)        # (n, 1) -> 

            # gradient with respect to M: 2/n * (feedforward) * input
            g_M = features.T @ g_f            # (x, n) @ (n, 1) = (x, 1)

            # gradient with respect to b: 2/n * (feedforward)
            g_b = g_f

            self.M -= learning_rate * g_M
            self.b -= learning_rate * g_b

if __name__ == "__main__": 
    # Test: y = 2x + 3x + 5
    x = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [2, 1], [5, 3]])
    y = np.array([8, 7, 10, 5, 12, 24]).reshape((6, 1))

    model = LinearRegression(2)

    print("Training =============================")
    model.train(x, y, 100, 0.1)

    print("Evaluate =============================")
    print("Average difference: ", model.eval(x, y))