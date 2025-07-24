import numpy as np
from sklearn import metrics
from datasets import load_mnist
from model import MLP
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = load_mnist()

model = MLP(X_train.shape[1], [512, 128, 32], 10)

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


for i in range(10):
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
y_pred_labels = np.argmax(y_pred, axis=1)

print(
    f"Best Loss: {best_test_loss: .6f}\nAccuracy at best Loss: {best_loss_acc: .2%}\nBest Accuracy: {best_acc: .2%}"
)

# Decision Boundary plot


# plot_loss(training_losses, test_losses)
# plot_loss_terminal(training_losses, test_losses)
# plot_predictions(X_test, y_pred)
# plot_predictions_terminal(X_test, y_pred)

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
plt.savefig("confusion_matrix.png")
