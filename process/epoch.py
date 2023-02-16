class epoch:
    def __init__(self, size):
        self.Z = []
        self.A = []
        self.A.append(self.X)
        self.L = 0  # initialize Loss function to be zero, for the entiretity of dataset.
        self.J = 0  # so initialize the cost function as well.
        self.size = size
        self.dW = list(range(len(self.size) - 1))  # Store the gradient of Weights against Loss function for each layer
        self.db = list(range(len(self.size) - 1))
