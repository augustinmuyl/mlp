# Multilayer Perceptron (MLP)

## Overview

This project is an implementation of a Multilayer Perceptron (MLP), a type of feedforward neural network, from scratch only using NumPy.

An MLP consists of layers of interconnected nodes (neurons), where each layer performs a linear transformation, followed by a non-linear activation function.

The goal of this project is to **build and train MLPs without using any deep learning libraries** (TensorFlow, PyTorch), to understand the internals of neural networks.

## Getting Started

Follow these steps to set up and run the project on your local machine:

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

### 3. Run the Model

```bash
python main.py
```

## ToDo

- [x] ~~Loss graph~~
- [x] ~~Predicted points plot~~
- [x] ~~Modularity~~
- [ ] Decision Boundary plot
- [ ] Optimize hyper parameters (grid search)
- [ ] CLI Tool
- [ ] More datasets (e.g. `make_classifications`)
