import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)

    def relu(self, x):
        return max(0, x)

    def deriv_relu(self, x):
        x = np.nan_to_num(x)
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def forward(self, X):
        Z1 = X @ self.W1
        self.A1 = self.relu(Z1)
        Z2 = self.A1 @ self.W2
        return self.sigmoid(Z2)

    def backward(self, y, y_pred):
        grad_W2 = self.A1.T * (y_pred - y)
