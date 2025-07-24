import argparse
import numpy as np
from sklearn import metrics
from model import MLP
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST or Fashion-MNIST")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion-mnist"],
        default="mnist",
        help="Which dataset to train on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Patience for early stopping",
    )
    return parser.parse_args()


def train_multiclass(X_train, X_test, y_train, y_test, y_train_oh, y_test_oh, args):
    model = MLP(X_train.shape[1], [512, 128, 32], 10)

    epochs = args.epochs
    best_test_loss = float("inf")
    best_acc = 0
    best_loss_acc = 0
    patience = args.patience
    count = 0
    lr = args.lr

    training_losses = []
    test_losses = []

    for i in range(epochs):
        # training
        y_pred = model.forward(X_train)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = -np.mean(np.sum(y_train_oh * np.log(y_pred), axis=1))
        training_losses.append(loss)
        model.backward(y_train_oh, y_pred, lr=lr)

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
    y_pred_labels = np.argmax(y_pred, axis=1)

    print(
        f"Best Loss: {best_test_loss: .6f}\nAccuracy at best Loss: {best_loss_acc: .2%}\nBest Accuracy: {best_acc: .2%}"
    )

    return model, y_test, y_pred_labels, training_losses, test_losses


def plot_confusion_matrix(y_test, y_pred_labels, dataset):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_labels, normalize="true", values_format=".2f", ax=ax
    )
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    for im in ax.get_images():
        im.set_interpolation("none")  # avoid antialiasing gaps

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"media/{dataset}_confusion_matrix.png")


def plot_loss_curve(training_losses, test_losses, dataset):
    plt.figure(figsize=(8, 4))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(f"media/{dataset}_loss_curve.png")


def plot_classification_examples(X, y_true, y_pred, dataset, correct=True, n=9):
    if correct:
        indices = np.where(y_pred == y_true)[0]
        title_prefix = "Correct"
    else:
        indices = np.where(y_pred != y_true)[0]
        title_prefix = "Incorrect"

    if len(indices) == 0:
        print("No matching examples to display.")
        return

    samples = np.random.choice(indices, size=min(n, len(indices)), replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, idx in zip(axes.ravel(), samples):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"T:{y_true[idx]}, P:{y_pred[idx]}")
        ax.axis("off")

    plt.suptitle(f"{title_prefix} Predictions — {dataset}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"media/{dataset}_{title_prefix.lower()}_examples.png")


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "mnist":
        dataset = "mnist"
        from datasets import load_mnist as load_data
    elif args.dataset == "fashion-mnist":
        dataset = "fashion_mnist"
        from datasets import load_fashion_mnist as load_data
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = load_data()

    model, y_test, y_pred_labels, training_losses, test_losses = train_multiclass(
        X_train, X_test, y_train, y_test, y_train_oh, y_test_oh, args
    )

    plot_confusion_matrix(y_test, y_pred_labels, dataset)
    plot_loss_curve(training_losses, test_losses, dataset)

    plot_classification_examples(X_test, y_test, y_pred_labels, dataset, correct=True)
    plot_classification_examples(X_test, y_test, y_pred_labels, dataset, correct=False)
