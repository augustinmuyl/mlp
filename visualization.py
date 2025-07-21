import os
import seaborn as sns
import plotext as plterm
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme()


def plot_loss(training_losses, test_losses):
    epochs = np.arange(len(training_losses))
    plt.figure(figsize=(8, 4))

    sns.lineplot(x=epochs, y=training_losses, label="Training Loss")
    sns.lineplot(x=epochs, y=test_losses, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_terminal(training_losses, test_losses):
    plterm.clear_figure()
    plterm.plot_size(70, 20)
    plterm.xfrequency(100)
    plterm.yfrequency(80)

    epochs = list(range(len(training_losses)))
    plterm.plot(epochs, training_losses, label="Train Loss")
    plterm.plot(epochs, test_losses, label="Test Loss")
    plterm.title("Loss over Epochs")
    plterm.xlabel("Epoch")
    plterm.ylabel("Loss")
    plterm.show()


def plot_predictions(X, y_pred, is_grid=False):
    y_class = np.rint(y_pred).flatten()

    plt.figure(figsize=(6, 6))

    if is_grid:
        Z = y_class.reshape(1000, 1000)
        x = np.linspace(-2, 3, 1000)
        y = np.linspace(-2, 2, 1000)
        plt.contourf(x, y, Z, levels=[-1, 0.5, 1], cmap="coolwarm", alpha=0.6)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y_class, cmap="coolwarm", edgecolors="k")

    plt.title("Predicted Class per Input Point")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_predictions_terminal(X, y_pred):
    plterm.clear_figure()
    plterm.plot_size(80, 25)

    y_class = np.rint(y_pred).flatten()

    # Split points by predicted class
    x_class0 = X[y_class == 0][:, 0]
    y_class0 = X[y_class == 0][:, 1]

    x_class1 = X[y_class == 1][:, 0]
    y_class1 = X[y_class == 1][:, 1]

    plterm.scatter(x_class0, y_class0, marker="x", label="Class 0")
    plterm.scatter(x_class1, y_class1, marker="o", label="Class 1")

    plterm.title("Predicted Class per Input Point")
    plterm.xlabel("Feature 1")
    plterm.ylabel("Feature 2")
    plterm.show()


def plot_last_loss():
    data = np.load("data/last_run/loss.npz")
    plot_loss(data["training"], data["test"])


def plot_last_loss_terminal():
    data = np.load("data/last_run/loss.npz")
    plot_loss_terminal(data["training"], data["test"])


def plot_last_predictions():
    data = np.load("data/last_run/predictions.npz")
    plot_predictions(data["X"], data["y_pred"])


def plot_last_predictions_terminal():
    data = np.load("data/last_run/predictions.npz")
    plot_predictions_terminal(data["X"], data["y_pred"])


def plot_last_decision_boundary():
    data = np.load("data/last_run/boundary.npz")
    plot_predictions(data["grid"], data["y_class"], is_grid=True)


def plot_last_decision_boundary_terminal():
    data = np.load("data/last_run/boundary.npz")
    plot_predictions_terminal(data["grid"], data["y_class"])
