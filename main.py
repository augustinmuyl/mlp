import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from model import MLP

X, y = make_moons(n_samples=1000, noise=0.2, random_state=None)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLP(X.shape[1], 10, 1)

epoch = 100

# Training
for i in range(epoch):
    y_pred = model.forward(X_train)
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = np.mean(-(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)))
    model.backward(y_train, y_pred)
    if i % 10 == 0:
        y_pred = model.forward(X_test)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = np.mean(-(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred)))
        print("\nTESTING:\n")
        print(loss)
        print(np.mean(y_test == np.rint(y_pred)))

# Testing
