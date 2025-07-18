import numpy as np

training_losses = []
test_losses = []

best_test_loss = float("inf")
best_loss_acc = 0
count = 0


def train_model(X_train, y_train, X_test, y_test, model, epoch, lr, patience=100):
    for i in range(epoch):
        # training
        y_pred = model.forward(X_train)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = np.mean(-(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)))
        training_losses.append(loss)
        model.backward(y_train, y_pred, lr)

        # testing
        y_pred = model.forward(X_test)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = np.mean(-(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred)))
        test_losses.append(loss)
        acc = np.mean(y_test == np.rint(y_pred))

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

        return training_losses, test_losses, y_pred
