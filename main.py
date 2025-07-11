import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from model import MLP
from visualization import plot_loss, plot_loss_terminal, plot_predictions, plot_predictions_terminal

X, y = make_moons(n_samples=10000, noise=0.2, random_state=None)
X = np.asarray(X)
y = np.asarray(y).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

model = MLP(X.shape[1], 80, 1)

epoch = 10000
best_test_loss = float("inf")
best_acc = 0
best_loss_acc = 0
patience = 100
count = 0

training_losses = []
test_losses = []

for i in range(epoch):
    # training

    y_pred = model.forward(X_train)
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = np.mean(-(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)))
    training_losses.append(loss)
    model.backward(y_train, y_pred)

    # testing

    y_pred = model.forward(X_test)
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = np.mean(-(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred)))
    test_losses.append(loss)
    acc = np.mean(y_test == np.rint(y_pred))

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

# plot_loss(training_losses, test_losses)
# plot_loss_terminal(training_losses, test_losses)
# plot_predictions(X_test, y_pred)
# plot_predictions_terminal(X_test, y_pred)
