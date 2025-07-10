import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def deriv_relu(self, x):
        x = np.nan_to_num(x)
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def deriv_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1
        self.A1 = self.relu(self.Z1)
        Z2 = self.A1 @ self.W2
        return self.sigmoid(Z2)

    def backward(self, y, y_pred, lr=1e-3):
        delta = (y_pred - y)
        grad_W2 = np.dot(delta.T, self.A1)
        grad_W1 = (np.dot(delta, self.W2.T) * self.deriv_relu(self.Z1)).T @ self.X
        self.W2 = self.W2 - lr * grad_W2.T
        self.W1 = self.W1 - lr * grad_W1.T
