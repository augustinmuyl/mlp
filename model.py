import numpy as np
import torch.nn as nn


class MLP:
    def __init__(
        self, input_dim: int, hidden_dims: list, output_dim: int, multiclass=False
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.multiclass = multiclass
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
        return 1 / (1 + np.exp(np.clip(-x, -709, 709)))

    def deriv_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        a = X
        self.Z = []
        self.A = [a]
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            z = a @ w + b
            self.Z.append(z)
            if i == len(self.W) - 1:
                if self.multiclass:
                    a = self.softmax(z)
                else:
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

        max_norm = 1.0

        for i in range(len(grad_W)):
            norm = np.linalg.norm(grad_W[i])
            if norm > max_norm:
                grad_W[i] *= max_norm / norm

        for i, (w, b) in enumerate(reversed(list(zip(self.W, self.B)))):
            w -= lr * grad_W[i].T
            b -= lr * grad_B[i]

    def save(self, path: str):
        np.savez(
            path,
            W=np.array(self.W, dtype=object),
            B=np.array(self.B, dtype=object),
            input_dim=self.input_dim,
            hidden_dims=np.array(self.hidden_dims),
            output_dim=self.output_dim,
        )

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        model = cls(
            input_dim=data["input_dim"],
            hidden_dims=data["hidden_dims"],
            output_dim=data["output_dim"],
            multiclass=False,
        )
        model.W = data["W"]
        model.B = data["B"]
        return model


class MLP_torch(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        """
        layers = [
            nn.Sequential(nn.Linear(in_features=dims[i], out_features=dims[i + 1]), nn.ReLU())
            for i in range(len(dims) - 2)
            if hidden_dims
        ]
        layers.append(
            nn.Sequential(
                nn.Linear(in_features=dims[len(dims) - 1], out_features=dims[len(dims)]),
                nn.Softmax(),
            )
        )
        self.model = nn.ModuleList(layers)
        """

        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(in_features=dims[i], out_features=dims[i + 1]), nn.ReLU())
                if i < len(dims) - 2
                else nn.Sequential(
                    nn.Linear(in_features=dims[i], out_features=dims[i + 1]),
                    nn.Softmax(dim=1),
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
