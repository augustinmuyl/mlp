import numpy as np
from sklearn.datasets import make_moons
from model import MLP

X, y = make_moons(n_samples=1000, noise=0.2, random_state=None)
y = y.reshape(-1, 1)

model = MLP(X.shape[1], 2, 1)

y_pred = model.forward(X)
model.backward(y, y_pred)
