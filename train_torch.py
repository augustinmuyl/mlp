import ssl
import numpy as np
from sklearn import metrics
from torch import nn
import torch
from datasets import load_mnist
from model import MLP_torch
from visualization import plot_loss, plot_loss_terminal, plot_predictions, plot_predictions_terminal
import matplotlib.pyplot as plt
from torch.optim import Adam


X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = load_mnist()

model = MLP_torch(X_train.shape[1], [], 10)

epoch = 10000
best_test_loss = float("inf")
best_acc = 0
best_loss_acc = 0
patience = 100
count = 0
lr = 1e-3

training_losses = []
test_losses = []

plot_preds = []
y_plot_preds = []

optim = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for i in range(epoch):
    # training
    optim.zero_grad()
    y_pred = model(torch.FloatTensor(X_train))
    loss = criterion(y_pred, torch.LongTensor(y_train))
    loss.backward()
    optim.step()
    training_losses.append(loss.item())

    # testing
    y_pred = model(torch.FloatTensor(X_test))
    loss = criterion(y_pred, torch.LongTensor(y_test))
    test_losses.append(loss.item())

    y_pred_labels = torch.argmax(y_pred, dim=1)
    acc = torch.mean((y_pred_labels == torch.LongTensor(y_test)).float())

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
        print(loss.item())
        print(acc.item())

print(
    f"Best Loss: {best_test_loss: .6f}\nAccuracy at best Loss: {best_loss_acc: .2%}\nBest Accuracy: {best_acc: .2%}"
)

# Decision Boundary plot

y_pred = model(torch.FloatTensor(X_test))
y_pred_labels = torch.argmax(y_pred, dim=1)

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
plt.savefig("media/torch_confusion_matrix.png")
