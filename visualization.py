import seaborn as sns
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
