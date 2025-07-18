import os
import numpy as np
from model import MLP
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from visualization import plot_loss_terminal, plot_predictions, plot_predictions_terminal
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


def train_model(hidden_layers, lr, epochs, patience=100):
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

    with Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn(" • loss={task.fields[loss]:.4f}"),
        TextColumn(" • acc={task.fields[acc]:.2%}"),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "[green]Training[/green]", total=epochs, loss=float("nan"), acc=float("nan")
        )

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

            progress.update(task, advance=1, loss=loss_test, acc=acc)

    y_pred_test = model.forward(X_test)

    output_dir = "data/last_run"
    os.makedirs(output_dir, exist_ok=True)

    np.savez(os.path.join(output_dir, "loss.npz"), training=training_losses, test=test_losses)
    np.savez(os.path.join(output_dir, "predictions.npz"), X=X_test, y_pred=y_pred_test)

    return best_test_loss, best_loss_acc
