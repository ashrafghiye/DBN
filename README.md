# Deep Belief Networks and Deep Neural Networks
Deep Learning II project

This project has two goals:

The First, is to implement a **Restricted Boltzman Machine** and a **Deep Belief Network** and use them as Generative Models.

Second, it aims to study the classification of handwritten digits using a randomly initialized DNN and another DNN pre-trained using DBN.


## Content

- Jupyter notebook `Numpy_Notebook.ipynb` which is a step by step guide of the study, all operations are coded using numpy from scratch.

- Jupyter notebook `Torch_Notebook.ipynb` is the exact same notebook but using torch tensors to benefit from Colab's GPU.

- The same code is organized in python scripts in `code`. 

- The resulted plots are saved in `images`.

- The necessary data are stored in `data`.

## Use

The folder `code` contains three class definitions `principal_RBM_alpha`, `principal_DBN_alpha`, `principal_DNN_MNIST`. These files can be imported directly in a main script and be used directly.

Also the file `utils.py` contains necessary and useful functions, such as two functions to load the MNIST and AlphaDigits datasets.

For example, here is a snipped code to load a dataset, train an RBM and start generating artificial examples:

```python
import principal_RBM_alpha as RBM
from utils import lire_alpha_digit

char = '3'
X_train = lire_alpha_digit(char)

p = X_train.shape[1]
q = 240
rbm = RBM.init_RBM(p,q)
rbm = RBM.train_RBM(rbm, n_epochs, learning_rate, batch_size, X_train)
RBM.generer_image_RBM(rbm, nb_images, nb_iterations)
```

Here is the kind of results you would get

![RBM_generated_char3](https://user-images.githubusercontent.com/24767888/113482714-33393f00-94a0-11eb-87ff-3431ebefc351.png)

Defining a pre-trained DNN for classification can be as simple as:

```python
import principal_DNN_MNIST as DNN
import principal_DBN_alpha as DBN
from utils import load_mnist

X_train, X_test, y_train, y_test = load_mnist()

p = X_train.shape[1]
q = y_train.shape[1]
neurons = [(p, 256), (256, 128), (128, q)]

dnn = DBN.init_DNN(num_layers, neurons)
dnn, _ = DBN.pretrain_DNN(dnn, n_epochs_rbm, lr, batch_size, X_train)
dnn = DNN.retropropagation(dnn, X_train, y_train, n_epochs_retro, lr, batch_size, "pre-trained")
```

To test the performance of your network, you can use the below function, which will plot the confusion matrix and return the accuracy and crossentropy on the test set.

```python
DNN.test_DNN(dnn, X_test, y_test, "Title")
```


## Analysis

The script `main.py` run one of three comparisons between two deep neural networks used to classify MNIST digits. The first neural was pre-trained in unsipervised way using Deep Belief Networks, and the second was randomly initialized.

Running this script will result in 2 images describing the accuracy and crossentropy on test set for both models for one of these cases:

1- Number of layers.


2- Number of hidden units.


3- Number of training examples.


For a detailed analysis, refer to the final report.


### Notes

- Everything is coded from scratch using numpy, including the backpropagation.

- The code is a mix of French (function names) and English (documentation and analysis).

- To test Deep Neural Networks, use torch notebook for efficient backpropagation.
