import os
import numpy as np
from model import MLP
from datasets import load_make_moons
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

BATCH_SIZE = 128


def train_model(hidden_layers, lr, epochs, patience, dynamic_epoch):
    X_train, X_test, y_train, y_test = load_make_moons()

    model = MLP(X_train.shape[1], hidden_layers, 1)

    best_test_loss = float("inf")
    best_loss_acc = 0
    count = 0

    training_losses = []
    test_losses = []

    num_batches = X_train.shape[0] // BATCH_SIZE

    if dynamic_epoch:
        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold green]Epoch {task.completed}[/bold green]"),
            TextColumn("• Test Loss: {task.fields[loss]:.4f}"),
            TextColumn("• Test Acc: {task.fields[acc]:.2%}"),
            TextColumn("• Patience Left: {task.fields[patience_left]}"),
            TimeElapsedColumn(),
        ]
        total = None
    else:
        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn(" • Test Loss: {task.fields[loss]:.4f}"),
            TextColumn(" • Test Acc: {task.fields[acc]:.2%}"),
            TimeRemainingColumn(),
        ]
        total = epochs

    with Progress(*progress_columns, transient=True, refresh_per_second=20) as progress:
        task = progress.add_task(
            "[green]Training[/green]",
            total=total,
            loss=float("nan"),
            acc=float("nan"),
            patience_left=patience,
        )

        for i in range(epochs):
            epoch_train_loss = 0

            for j in range(num_batches):
                start = j * BATCH_SIZE
                end = start + BATCH_SIZE
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # training
                y_pred = model.forward(X_batch)
                y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
                loss = -np.mean(np.sum(y_batch * np.log(y_pred), axis=1))
                model.backward(y_batch, y_pred, lr=lr)
                epoch_train_loss += loss

            epoch_train_loss /= num_batches
            training_losses.append(epoch_train_loss)

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
            if acc == 1:
                break

            update_fields = {"loss": loss_test, "acc": acc}
            if dynamic_epoch:
                update_fields["patience_left"] = patience - count
            progress.update(task, advance=1, **update_fields)

    y_pred_test = model.forward(X_test)

    output_dir = "data/last_run"
    os.makedirs(output_dir, exist_ok=True)

    np.savez(os.path.join(output_dir, "loss.npz"), training=training_losses, test=test_losses)
    np.savez(os.path.join(output_dir, "predictions.npz"), X=X_test, y_pred=y_pred_test)

    # Decision boundary calculation

    x = np.linspace(-2, 3, 1000)
    y = np.linspace(-2, 2, 1000)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    y_pred = model.forward(grid)
    y_class = np.rint(y_pred).flatten()

    np.savez(os.path.join(output_dir, "boundary.npz"), grid=grid, y_class=y_class)

    return best_test_loss, best_loss_acc, i
