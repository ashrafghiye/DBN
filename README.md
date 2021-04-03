# Deep Belief Networks and Deep Neural Networks
Deep Learning II project: DBN-DNN

This project has two goals

First, it aims to implement a **Restricted Boltzman Machine** and a **Deep Belief Network** and use them as Generative Models.

Second, it aims to study the classification of handwritten digits using a randomly initialized DNN and another DNN pre-trained using DBN.


## Content

- Jupyter notebook `DL2.ipynb` which is a step by step guide of the study. 

- The same code is organized in python scripts in `code'. 

- The resulted plots are saved in `images`.

- The necessary data are stored in `data`.

## Use

The folder `code` contains three class definitions `principal_RBM_alpha`, `principal_DBN_alpha`, `principal_DNN_MNIST`. These files can be imported directly in a main script and be used directly.

Also the file `utils.py` contains necessary and useful functions, such as two functions to load the MNIST and AlphaDigits datasets.

For example, here is a snipped code to load a dataset, train an RBM and start generating artificial examples:

```python
import principal_RBM_alpha as RBM
from utils import lire_alpha_digit

X_train = lire_alpha_digit('3')

p = X_train.shape[1]
q = 240
rbm = RBM.init_RBM(p,q)
rbm = RBM.train_RBM(rbm, n_epochs, learning_rate, batch_size, X_train)
RBM.generer_image_RBM(rbm, nb_images, nb_iterations)
```
