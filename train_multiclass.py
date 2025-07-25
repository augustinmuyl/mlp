import argparse
import numpy as np
from sklearn import metrics
from model import MLP
import matplotlib.pyplot as plt

FASHION_LABELS = [
    "T-shirt/top",  # 0
    "Trouser",  # 1
    "Pullover",  # 2
    "Dress",  # 3
    "Coat",  # 4
    "Sandal",  # 5
    "Shirt",  # 6
    "Sneaker",  # 7
    "Bag",  # 8
    "Ankle boot",  # 9
]

MNIST_LABELS = [str(i) for i in range(10)]

BATCH_SIZE = 128


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
        default=1e-4,
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
    model = MLP(X_train.shape[1], [256, 128], 10, multiclass=True)

    epochs = args.epochs
    best_test_loss = float("inf")
    best_acc = 0
    best_loss_acc = 0
    patience = args.patience
    count = 0
    lr = args.lr

    training_losses = []
    test_losses = []

    num_batches = X_train.shape[0] // BATCH_SIZE

    for i in range(epochs):
        epoch_train_loss = 0

        for j in range(num_batches):
            start = j * BATCH_SIZE
            end = start + BATCH_SIZE
            X_batch = X_train[start:end]
            y_batch = y_train_oh[start:end]

            # training
            y_pred = model.forward(X_batch)
            y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
            loss = -np.mean(np.sum(y_batch * np.log(y_pred), axis=1))
            model.backward(y_batch, y_pred, lr=lr)
            epoch_train_loss += loss

        epoch_train_loss /= num_batches
        training_losses.append(epoch_train_loss)
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

    return model, y_test, y_pred, y_pred_labels, training_losses, test_losses


def plot_confusion_matrix(y_test, y_pred_labels, dataset, label_map):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_labels,
        display_labels=label_map,
        normalize="true",
        values_format=".2f",
        ax=ax,
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


def plot_classification_examples(
    X, y_true, y_pred, y_pred_labels, dataset, label_map, correct=True, n=9
):
    confidences = np.max(y_pred, axis=1)
    if correct:
        indices = np.where(y_pred_labels == y_true)[0]
        title_prefix = "Correct"
        sort_order = confidences
    else:
        indices = np.where(y_pred_labels != y_true)[0]
        title_prefix = "Incorrect"
        sort_order = -confidences

    if len(indices) == 0:
        print("No matching examples to display.")
        return

    sorted_indices = indices[np.argsort(sort_order[indices])]
    samples = sorted_indices[: min(n, len(sorted_indices))]

    grid_size = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))

    for ax, idx in zip(axes.ravel(), samples):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(
            f"T:{label_map[y_true[idx]]}, P:{label_map[y_pred_labels[idx]]}\n{confidences[idx] * 100:.1f}%",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes.ravel()[len(samples) :]:
        ax.axis("off")

    plt.suptitle(f"{title_prefix} Predictions — {dataset}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"media/{dataset}_{title_prefix.lower()}_examples.png")


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "mnist":
        dataset = "mnist"
        label_map = MNIST_LABELS
        from datasets import load_mnist as load_data
    elif args.dataset == "fashion-mnist":
        dataset = "fashion_mnist"
        label_map = FASHION_LABELS
        from datasets import load_fashion_mnist as load_data
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = load_data()

    model, y_test, y_pred, y_pred_labels, training_losses, test_losses = train_multiclass(
        X_train, X_test, y_train, y_test, y_train_oh, y_test_oh, args
    )

    plot_confusion_matrix(y_test, y_pred_labels, dataset, label_map)
    plot_loss_curve(training_losses, test_losses, dataset)

    plot_classification_examples(
        X_test, y_test, y_pred, y_pred_labels, dataset, label_map, correct=True
    )
    plot_classification_examples(
        X_test, y_test, y_pred, y_pred_labels, dataset, label_map, correct=False
    )
