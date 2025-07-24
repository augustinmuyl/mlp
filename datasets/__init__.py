import numpy as np
from sklearn.datasets import make_moons, fetch_openml
from sklearn.model_selection import train_test_split


def load_make_moons():
    X, y = make_moons(n_samples=10000, noise=0.2, random_state=None)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train, X_test, y_train, y_test


def load_mnist():
    # ssl._create_default_https_context = ssl._create_unverified_context

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    y_train_oh = np.eye(10)[y_train]
    y_test_oh = np.eye(10)[y_test]

    return X_train, X_test, y_train, y_test, y_train_oh, y_test_oh


def load_fashion_mnist():
    # ssl._create_default_https_context = ssl._create_unverified_context

    mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    y_train_oh = np.eye(10)[y_train]
    y_test_oh = np.eye(10)[y_test]

    return X_train, X_test, y_train, y_test, y_train_oh, y_test_oh
