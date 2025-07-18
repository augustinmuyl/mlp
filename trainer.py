import numpy as np
from model import MLP
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from visualization import plot_loss_terminal, plot_predictions_terminal


def train_model(hidden_layers, lr, epochs, use_terminal_plot, patience=100):
    X, y = make_moons(n_samples=10000, noise=0.2, random_state=None)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    model = MLP(X.shape[1], hidden_layers, 1)

    best_test_loss = float("inf")
    best_loss_acc = 0
    count = 0

    training_losses = []
    test_losses = []

    for i in range(epochs):
        # Training
        y_pred_train = model.forward(X_train)
        y_pred_train = np.clip(y_pred_train, 1e-9, 1 - 1e-9)
        loss_train = np.mean(
            -(y_train * np.log(y_pred_train) + (1 - y_train) * np.log(1 - y_pred_train))
        )
        training_losses.append(loss_train)

        model.backward(y_train, y_pred_train, lr)

        # Testing
        y_pred_test = model.forward(X_test)
        y_pred_test = np.clip(y_pred_test, 1e-9, 1 - 1e-9)
        loss_test = np.mean(
            -(y_test * np.log(y_pred_test) + (1 - y_test) * np.log(1 - y_pred_test))
        )
        test_losses.append(loss_test)
        acc = np.mean(y_test == np.rint(y_pred_test))

        if loss_test <= best_test_loss:
            best_test_loss = loss_test
            best_loss_acc = acc
            count = 0
        else:
            count += 1
            if count == patience:
                break

        if use_terminal_plot:
            plot_loss_terminal(training_losses, test_losses)

        return best_test_loss, best_loss_acc
