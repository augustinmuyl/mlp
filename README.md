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

## To-Do

- [ ] Save and load models
- [ ] Optimize hyper parameters (grid search)
- [ ] More datasets (e.g. `make_classifications`)
