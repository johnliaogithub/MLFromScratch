import numpy as np

def sigmoid(z: np.array): 
    """
        \sigma(z) = \frac{1}{1 + e^{-z}}
                  = (1 + e^{-z}) ^ {-1}
    """
    return np.reciprocal(1 + np.exp(-z))

def g_sigmoid(z: np.array):
    """
        \sigma'(z) = -1 * (1 + e^{-z}) ^ -2 * (e^{-z} * -1)
                   = \frac{e^{-z}}{(1 - e^{-z})^2}
                   = \sigma(z) * (1 - \sigma(z))
    """
    return sigmoid(z) * (1 - sigmoid(z))

def binary_cross_entropy(pred: np.array, targ: np.array):
    """
    y_i is actual, p_i is predicted

    - \frac{1}{N} \sum{y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i)}
    """
    assert(pred.shape == targ.shape)

    return -np.average(targ * np.log(pred) + (1 - targ) * np.log(1 - pred))

def g_BCE(pred: np.array, targ: np.array): 
    """
    gradient with respect to pred

    Recall: ln'(x) = 1/x

    - \frac{1}{N} \sum{y_i / p_i - (1 - y_i) / (1 - p_i)}
    """
    return -np.average((targ / pred) - ((1 - targ) / (1 - pred)))

class LogisticRegression: 
    def __init__(self, input_dimensions: int):      # x
        assert(input_dimensions >= 0)

        self.input_dimensions = input_dimensions

        # initialize state
        self.M = np.random.randn(input_dimensions).reshape((input_dimensions, 1))  # (x, 1)
        self.b = np.random.randn()                  # (1, )

    def eval(self, features: np.array, targets: np.array):
        return np.average(np.abs(self.feedforward(features) - targets))
    
    def linear(self, features: np.array): 
        return features @ self.M + self.b

    def feedforward(self, features: np.array): 
        return sigmoid(features @ self.M + self.b)           # (n, 1)

    def train(self, features: np.array, targets: np.array, epochs:int = 0, learning_rate:float = 0.01): 
        assert(len(features.shape) == len(targets.shape) == 2)
        assert(features.shape == (targets.size, self.input_dimensions))

        n = features.shape[0]

        for i in range(epochs): 
            feedforward_1 = self.linear(features)
            feedforward_2 = self.feedforward(features)
            
            bce = binary_cross_entropy(feedforward_2, targets)

            print("Training loop: {}, BCE: {}".format(i, bce))

            # gradient of feedforward
            g_J = g_BCE(feedforward_2, targets)        # (n, 1) -> 
            g_z = g_J * g_sigmoid(feedforward_1)

            # gradient with respect to M: 2/n * (feedforward) * input
            g_M = features.T @ g_z            # (x, n) @ (n, 1) = (x, 1)

            # gradient with respect to b: 2/n * (feedforward)
            g_b = g_z

            self.M -= learning_rate * g_M
            self.b -= learning_rate * g_b

if __name__ == "__main__": 
    # Test: y = 2x + 3x + 5, 1 if below 10
    x = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [2, 1], [5, 3]])
    y = np.array([1, 1, 0, 1, 0, 0]).reshape((6, 1))

    model = LogisticRegression(2)

    print("Training =============================")
    model.train(x, y, 1000, 0.1)

    print("Evaluate =============================")
    print("Predictions: ", model.feedforward(x))
    print("Average difference: ", model.eval(x, y))

    # bug: returned nan twice