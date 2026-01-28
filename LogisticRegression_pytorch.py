import torch

EPSILON = 1e-15

def sigmoid(z: torch.tensor): 
    """
        \\sigma(z) = \\frac{1}{1 + e^{-z}}
                  = (1 + e^{-z}) ^ {-1}
    """
    return torch.reciprocal(1 + torch.exp(-z))

def binary_cross_entropy(pred: torch.tensor, targ: torch.tensor):
    """
    y_i is actual, p_i is predicted

    - \\frac{1}{N} \\sum{y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i)}
    """
    assert(pred.shape == targ.shape)

    pred = torch.clip(pred, EPSILON, 1 - EPSILON)
    return -torch.mean(targ * torch.log(pred) + (1 - targ) * torch.log(1 - pred))

class LogisticRegression: 
    def __init__(self, input_dimensions: int):      # x
        assert(input_dimensions >= 0)

        self.input_dimensions = input_dimensions

        # initialize state
        self.M = torch.randn((input_dimensions, 1), requires_grad=True)  # (x, 1)
        self.b = torch.randn((1, ), requires_grad=True)                  # (1, )

    def evaluate_accuracy(self, features: torch.tensor, targets: torch.tensor):
        """
        evaluates the model performance based on accuracy
      
        :param features: input features
        :type features: torch.tensor
        :param targets: targets
        :type targets: torch.tensor

        accuracy = true / total
        """
        pred = self.feedforward(features) > 0.5
        classes = pred == targets
        return torch.mean(classes.to(torch.float32))
    
    def feedforward(self, features: torch.tensor): 
        return sigmoid(features @ self.M + self.b)           # (n, 1)

    def fit(self, features: torch.tensor, targets: torch.tensor, epochs:int = 0, learning_rate:float = 0.01): 
        """
        Train the linear regression model
        
        :param features: input data to train on.
        :type features: torch.tensor
        :param targets: target data to train on.
        :type targets: torch.tensor
        :param epochs: number of training loops
        :type epochs: int
        :param learning_rate: learning rate of the model. Coefficient of the gradients when updating parameters. 
        :type learning_rate: float

        In the math: 
            - x: number of inputs
            - n: number of training data
        """
        assert(len(features.shape) == len(targets.shape) == 2)
        #assert(features.shape == (targets.size(), self.input_dimensions))

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

            self.M.grad = None
            self.b.grad = None

            bce.backward()
            
            with torch.no_grad():
                self.M -= learning_rate * self.M.grad
                self.b -= learning_rate * self.b.grad


if __name__ == "__main__": 
    # Test: y = 2x + 3x + 5, 1 if below 10
    x = torch.Tensor([[0, 1], [1, 0], [1, 1], [0, 0], [2, 1], [5, 3]])
    y = torch.Tensor([1, 1, 0, 1, 0, 0]).reshape((6, 1))

    model = LogisticRegression(2)

    print("Training =============================")
    model.fit(x, y, 100, 0.1)

    print("Evaluate =============================")
    print("Predictions: {}".format(model.feedforward(x).detach().numpy()))
    print("Accuracy ", model.evaluate_accuracy(x, y))