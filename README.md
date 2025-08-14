# Multilayer Perceptron (MLP)

## Overview

This project is an implementation of a Multilayer Perceptron (MLP), a type of feedforward neural network, from scratch only using NumPy.

An MLP consists of layers of interconnected nodes (neurons), where each layer performs a linear transformation, followed by a non-linear activation function.

The goal of this project is to **build and train MLPs without using any deep learning libraries** (TensorFlow, PyTorch), to understand the internals of neural networks.

## Getting Started

Follow these steps to set up the project on your local machine:

### 1. Clone the Repo

```bash
git clone https://github.com/augustinmuyl/mlp.git
cd mlp
```

### 2. Setup Python

```bash
python3 -m venv .venv  # create venv
source .venv/bin/activate  # activate venv
pip install -r requirements.txt  # install dependencies
```

## Usage

To launch the interactive CLI:

```bash
python cli.py
```

You can train a model using your preferred parameters for the following:

- The number of hidden layers and neurons per layer
- The learning rate
- Either:
  - A fixed number of epochs, or
  - Enable **dynamic epoch stopping** using a "patience" value, which stops training early when the model stops improving 

After training, the CLI will automatically save model and training info to `data/last_run/`

You can then access the following plots:

- Training loss vs Test loss
- Predicted points
- Decision boundary

> Note: all plots have both terminal and GUI versions

## Features

- Train MLPs from scratch using NumPy
- Interactive CLI to configure training
- Terminal and GUI plots:
  - Loss curves
  - Predictions
  - Decision boundary
- Dynamic epoch stopping with configurable patience

## 📊 Results

### MNIST

- Best Accuracy: **98.01%**

- Confusion Matrix:

  ![mnist_confusion_matrix](media/mnist_confusion_matrix.png)

- Loss Curve:

  ![mnist_loss_curve](media/mnist_loss_curve.png)

- Correct Examples:

  ![mnist_correct_examples](media/mnist_correct_examples.png)

- Incorrect Examples:

  ![mnist_incorrect_examples](media/mnist_incorrect_examples.png)

### Fashion-MNIST

- Best Accuracy: **88.80%**

- Confusion Matrix:

  ![fashion_mnist_confusion_matrix](media/fashion_mnist_confusion_matrix.png)

- Loss Curve:

  ![fashion_mnist_loss_curve](media/fashion_mnist_loss_curve.png)

- Correct Examples:

  ![fashion_mnist_correct_examples](media/fashion_mnist_correct_examples.png)

- Incorrect Examples:

  ![fashion_mnist_incorrect_examples](media/fashion_mnist_incorrect_examples.png)

## Mathematical Derivations of Backpropagation

### Sigmoid

$$
\begin{align*}
    \sigma (z)&=\frac{1}{1+e^{-z}} \\
    \frac{\partial \sigma}{\partial z}&=\frac{e^{-z}}{(1+e^{-z})^2} \\
    &=\frac{1}{1+e^{-z}}\cdot \frac{e^{-z}}{1+e^{-z}} \\
    &=\sigma (z) \cdot \frac{e^{-z}}{1+e^{-z}} \\
    &=\sigma (z) \cdot \frac{1+e^{-z}-1}{1+e^{-z}} \\
    &\mathbf{=\sigma (z)(1-\sigma (z))}
\end{align*}
$$

### Binary Cross-Entropy Loss

## 🛠️ Remaining Tasks (Roadmap)

### 🔄 PyTorch Comparison

- [ ] Rename `test.py` → `compare_pytorch.py`
- [ ] Show NumPy vs PyTorch results (accuracy, loss)
- [ ] Optional: compare training time
- [ ] Summarize in a table in the README

---

### 🔍 Grid Search (Optional)

- [ ] Create `grid_search.py`
- [ ] Run combinations of hidden layers and learning rates
- [ ] Save best config and scores
- [ ] (Optional) Include results in README

---

### 📄 README Improvements

- [ ] Add **Performance** section with table:
  - [ ] make_moons, MNIST, Fashion-MNIST
  - [ ] NumPy vs PyTorch accuracy
- [ ] Add screenshots or plot images
- [ ] Document CLI features: dataset selection, saving/loading, plots
- [ ] Highlight educational vs practical tradeoffs

---

### 🌐 Portfolio Integration

- [ ] Create CLI demo `.gif` (e.g. using `asciinema`)
- [ ] Add LinkedIn/GitHub description (1–2 line summary)
- [ ] Add tags and project topics to GitHub
