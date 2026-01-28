import numpy as np

EPSILON = 1e-15

def sigmoid(z: np.array): 
    """
        \\sigma(z) = \\frac{1}{1 + e^{-z}}
                  = (1 + e^{-z}) ^ {-1}
    """
    return np.reciprocal(1 + np.exp(-z))

def binary_cross_entropy(pred: np.array, targ: np.array):
    """
    y_i is actual, p_i is predicted

    - \\frac{1}{N} \\sum{y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i)}
    """
    assert(pred.shape == targ.shape)

    pred = np.clip(pred, EPSILON, 1 - EPSILON)
    return -np.mean(targ * np.log(pred) + (1 - targ) * np.log(1 - pred))

class LogisticRegression: 
    def __init__(self, input_dimensions: int):      # x
        assert(input_dimensions >= 0)

        self.input_dimensions = input_dimensions

        # initialize state
        self.M = np.random.randn(input_dimensions).reshape((input_dimensions, 1))  # (x, 1)
        self.b = np.random.randn()                  # (1, )

    def eval(self, features: np.array, targets: np.array):
        """
        evaluates the model performance based on accuracy
        
        :param features: input features
        :type features: np.array
        :param targets: targets
        :type targets: np.array

        accuracy = true / total
        """
        pred = self.feedforward(features) > 0.5
        return np.mean(pred == targets)
    
    def feedforward(self, features: np.array): 
        return sigmoid(features @ self.M + self.b)           # (n, 1)

    def train(self, features: np.array, targets: np.array, epochs:int = 0, learning_rate:float = 0.01): 
        """
        Train the linear regression model
        
        :param features: input data to train on.
        :type features: np.array
        :param targets: target data to train on.
        :type targets: np.array
        :param epochs: number of training loops
        :type epochs: int
        :param learning_rate: learning rate of the model. Coefficient of the gradients when updating parameters. 
        :type learning_rate: float

        In the math: 
            - x: number of inputs
            - n: number of training data
        """
        assert(len(features.shape) == len(targets.shape) == 2)
        assert(features.shape == (targets.size, self.input_dimensions))

        n = features.shape[0]

        for i in range(epochs): 
            feedforward = self.feedforward(features)
            
            bce = binary_cross_entropy(feedforward, targets)

            print("Training loop: {}, BCE: {}".format(i, bce))
        

            # BACKPROPAGATION
            """
            calculating the gradient of the loss function with respect to z
              - \\frac{1}{N} \\sum{y_i / p_i - (1 - y_i) / (1 - p_i)} * \\sigma(z) * (1 - \\sigma(z))
            = \\frac{p_i - y_i}{N}
            """

            d_z = feedforward - targets       # (n, 1)

            # gradient with respect to M: 2/n * (feedforward) * input
            g_W = features.T @ d_z / n            # (x, n) @ (n, 1) = (x, 1)

            # gradient with respect to b: 2/n * (feedforward)
            g_b = np.sum(d_z) / n             # (1, )

            self.M -= learning_rate * g_W
            self.b -= learning_rate * g_b

if __name__ == "__main__": 
    # Test: y = 2x + 3x + 5, 1 if below 10
    x = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [2, 1], [5, 3]])
    y = np.array([1, 1, 0, 1, 0, 0]).reshape((6, 1))

    model = LogisticRegression(2)

    print("Training =============================")
    model.train(x, y, 100, 0.1)

    print("Evaluate =============================")
    print("Predictions: {}".format(model.feedforward(x)))
    print("Accuracy ", model.eval(x, y))