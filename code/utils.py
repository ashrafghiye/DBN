import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

import scipy
import os

import wget

datapath = '../data/'

def load_mnist():
    """
    This function import the mnist dataset which consists of 70,000 images of size 28*28.
    train size: 60,000
    test size: 10,000

    Each image is represented as a row of size 784 (flattend).
    Note that we will threshold to get binary images, and we will one-hot encode the labels to obtain y as a matrix of size n_samples*n_classes
    Reference: http://yann.lecun.com/exdb/mnist/
    It returns the training images, testing images and their respect labels.
    """
    mnist = fetch_openml('mnist_784')

    # Trunk the data
    n_train = 60000
    n_test = 10000

    # Define training and testing sets
    indices = np.arange(len(mnist.data))

    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train + 1, n_train + n_test)

    X_train, y_train = mnist.data[train_idx], mnist.target[train_idx].astype(int)
    X_test, y_test = mnist.data[test_idx], mnist.target[test_idx].astype(int)

    # Binarization
    X_train[X_train > 0] = 1
    X_test[X_test > 0] = 1

    # One-hot encoding
    n_values = np.max(y_train) + 1

    y_train_o = np.eye(n_values)[y_train]
    y_test_o = np.eye(n_values)[y_test]

    return X_train, X_test, y_train_o, y_test_o


def plot_digit(x, y, dataset='mnist'):
    """
    x: 1D np.array -> one flattend digit
    y: one-hot encoded label
    dataset: type of dataset used. mnist or alphadigit
    """
    assert dataset in ['mnist', 'alphadigit']
    n = -1

    if dataset == 'mnist':
        n = 28
        label = np.argmax(y)
    else:
        n = 20
        label = y

    plt.figure()
    if dataset == 'mnist' or y.isdigit():
        plt.title("digit {0}".format(label))
    else:
        plt.title("character {0}".format(label))
    plt.imshow(x.reshape(n, -1), cmap='gray')
    plt.axis('off')
    plt.show()

def get_char_ord(char='0'):
    """
    Get the order of character..
    '0-9' -> 0-9, 'A' -> 10, 'B' -> 11, ..., 'Z' -> 35
    """
    assert type(char) is str

    if char.isdigit():
        return int(char)
    else:
        return ord(char) - ord('A') + 10

def lire_alpha_digit(char='0'):
    """
    import the AlphaDigits dataset

    It consists of 20x16 Binary digits of "0" through "9" and capital "A" through "Z".
    39 examples of each class.

    Only return the 39 examples corresponding the argument char.
    char: str "0" through "9" and capital "A" through "Z"


    reference: https://cs.nyu.edu/~roweis/data.html
    """

    file_name = datapath+"binaryalphadigs.mat"
    output_directory = datapath
    url = 'https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat'

    if not os.path.isfile(file_name):  # check whether the file exist or not
        wget.download(url, out=output_directory)

    X = pd.DataFrame(scipy.io.loadmat(file_name)['dat'])

    n_samples = len(X) * 39
    IMG = pd.DataFrame(np.zeros((n_samples, 20 * 16), dtype=np.int8))

    ligne = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            IMG.iloc[ligne, :] = np.concatenate([X.iloc[i, j][k] for k in range(20)])
            ligne += 1

    indx_char = get_char_ord(char)

    return np.array(IMG)[39 * indx_char:39 * (indx_char + 1)]


def sigmoid(x):
  return (1 / (1 + np.exp(-x)))


def plot_examples_alphadigits(X_train, x_generated):
    """
    takes as input X_train (alphadigits stacked into rows) and x_generated a tensor of dimensiosn n_images x 20 x 16
    plots a comparaison between generated and training examples..
    """
    fig = plt.figure(figsize=(8, 8))
    n_generated = x_generated.shape[0]
    columns = min(n_generated, 4)
    rows = 2
    samples = np.random.choice(np.arange(X_train.shape[0]), size=columns, replace=False)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i > columns:
            plt.imshow(X_train[i].reshape(20, 16), cmap='gray')
            plt.title('ex {0}'.format(samples[i - columns - 1]))
        else:
            plt.imshow(x_generated[i - 1], cmap='gray')
            plt.title('generated ex {0}'.format(i))

        plt.axis('off')

    plt.show()