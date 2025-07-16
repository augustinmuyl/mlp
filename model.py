import numpy as np


class MLP:
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        self.W = []
        self.B = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.W.append(np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2 / dims[i]))
            self.B.append(np.zeros((1, dims[i + 1])))

    def relu(self, x):
        return np.maximum(0, x)

    def deriv_relu(self, x):
        x = np.nan_to_num(x)
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x + 1e-9))

    def deriv_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, X):
        a = X
        self.Z = []
        self.A = [a]
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            z = a @ w + b
            self.Z.append(z)
            if i == len(self.W) - 1:
                a = self.sigmoid(z)
            else:
                a = self.relu(z)
            self.A.append(a)
        return a

    def backward(self, y, y_pred, lr=1e-3):
        delta = y_pred - y
        grad_W = []
        grad_B = []

        for i in reversed(range(len(self.W))):
            grad_W.append(np.dot(delta.T, self.A[i]))
            grad_B.append(np.sum(delta, 0, keepdims=True))

            if i != 0:
                delta = np.dot(delta, self.W[i].T) * self.deriv_relu(self.Z[i - 1])

        for i, (w, b) in enumerate(reversed(list(zip(self.W, self.B)))):
            w -= lr * grad_W[i].T
            b -= lr * grad_B[i]
