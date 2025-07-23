import ssl
import numpy as np
from sklearn.datasets import make_moons, fetch_openml
from sklearn.model_selection import train_test_split
from model import MLP
from visualization import plot_loss, plot_loss_terminal, plot_predictions, plot_predictions_terminal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

# X, y = make_moons(n_samples=10000, noise=0.2, random_state=None)
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

y_train_oh = np.eye(10)[y_train]
y_test_oh = np.eye(10)[y_test]

model = MLP(X.shape[1], [512, 128, 32], 10)

epoch = 10000
best_test_loss = float("inf")
best_acc = 0
best_loss_acc = 0
patience = 100
count = 0

training_losses = []
test_losses = []

plot_preds = []
y_plot_preds = []


for i in range(epoch):
    # training
    y_pred = model.forward(X_train)
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = -np.mean(np.sum(y_train_oh * np.log(y_pred), axis=1))
    training_losses.append(loss)
    model.backward(y_train_oh, y_pred, 1e-5)

    # testing
    y_pred = model.forward(X_test)
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = -np.mean(np.sum(y_test_oh * np.log(y_pred), axis=1))
    test_losses.append(loss)

    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = np.mean(y_pred_labels == y_test)

    if acc >= best_acc:
        best_acc = acc
    if loss <= best_test_loss:
        best_test_loss = loss
        best_loss_acc = acc
        count = 0
    else:
        count += 1
        if count == patience:
            print(f"Early stopping (epoch: {i + 1})...")
            break

    if i % 10 == 0:
        print("\nTESTING:\n")
        print(loss)
        print(acc)
y_pred = model.forward(X_test)

print(
    f"Best Loss: {best_test_loss: .6f}\nAccuracy at best Loss: {best_loss_acc: .2%}\nBest Accuracy: {best_acc: .2%}"
)

# Decision Boundary plot


# plot_loss(training_losses, test_losses)
# plot_loss_terminal(training_losses, test_losses)
# plot_predictions(X_test, y_pred)
plot_predictions_terminal(X_test, y_pred)
