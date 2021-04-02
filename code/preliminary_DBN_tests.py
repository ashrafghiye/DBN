"""
This script evaluate the performance of a DBN network.

It first defines two functions to help evaluate the learning process (quantitatevely) with a learning curve.
Also, it plots some generated examples to assess the power of network in learning the distribution of characters.

Results are saved in ../images/DBN_analysis
"""

import utils
import principal_DBN_alpha as DBN
import matplotlib.pyplot as plt
import numpy as np



imagepath = '../images/DBN_analysis/'


def DBN_quant_analysis(char='3', num_layers=3, neurons=None, lr=0.2, n_epochs=200, batch_size=6, layer=0, visualize=True, outputpath = imagepath):
    """
    Plot the reconstractive error after training with the specified hyperparamters for a given character
    """

    X_train = utils.lire_alpha_digit(char)
    p = X_train.shape[1]
    if neurons is None:
        neurons = [(p, p // 2), (p // 2, p // 4), (p // 4, p // 6)]

    assert num_layers == len(neurons)

    dbn = DBN.init_DNN(num_layers, neurons)

    dbn, err_eqm = DBN.pretrain_DNN(dbn, n_epochs, lr, batch_size, X_train)

    if visualize:
        plt.figure()
        plt.plot(range(n_epochs), err_eqm[layer])
        plt.title('EQM: lr={0}, bs={1}, epochs={2}, neruons={3}, char={4}, layer={5}'.format(lr, batch_size, n_epochs,
                                                                                             "/".join(
                                                                                                 [str(n[1]) for n in
                                                                                                  neurons]), char,
                                                                                             layer))
        plt.xlabel('epochs')
        plt.ylabel('EQM')
        if imagepath.split('/')[-1] != "DBN_analysis":
            plt.savefig('{0}.png'.format(outputpath))
        plt.show()

    return err_eqm

def DBN_qualt_analysis(dbn, X_train, nb_images=3, nb_iterations=1000, outputpath = imagepath):
    """
    Plot some examples after training with the specified hyperparamters for a given character
    """
    x_generated_dbn = DBN.generer_image_DBN(dbn, nb_images, nb_iterations, visualize=False)
    utils.plot_examples_alphadigits(X_train, x_generated_dbn, nb_iterations, outputpath)


#initial parameters
X_train = utils.lire_alpha_digit('0')
p = X_train.shape[1]
q = 240
lr = 0.15
n_epochs= 100
batch_size = 9
num_layers = 3
neurons = [(p, 2*p//3), (2*p//3, p//2), (p//2,p//3)]


## Learning curve
char = '3'
layer = 0
DBN_quant_analysis(char, num_layers, neurons, lr, n_epochs, batch_size, layer, True, imagepath+"DBN_eqm_char{0}_layer{1}".format(char, layer));
layer = 2
DBN_quant_analysis(char, num_layers, neurons, lr, n_epochs, batch_size, layer, True, imagepath+"DBN_eqm_char{0}_layer{1}".format(char, layer));
char = 'C'
DBN_quant_analysis(char, num_layers, neurons, lr, n_epochs, batch_size, layer, True, imagepath+"DBN_eqm_char{0}_layer{1}".format(char, layer));
layer = 2
DBN_quant_analysis(char, num_layers, neurons, lr, n_epochs, batch_size, layer, True, imagepath+"DBN_eqm_char{0}_layer{1}".format(char, layer));


## Qualitative analysis

nb_images, nb_iterations = 3, 1000

char='3'
X_train = utils.lire_alpha_digit(char)
p = X_train.shape[1]

dbn = DBN.init_DNN(num_layers, neurons)
dbn, err_eqm = DBN.pretrain_DNN(dbn, n_epochs, lr, batch_size, X_train)

DBN_qualt_analysis(dbn, X_train, nb_images, nb_iterations, imagepath+"DBN_generated_char{0}".format(char) )

char='C'
X_train = utils.lire_alpha_digit(char)
p = X_train.shape[1]

dbn = DBN.init_DNN(num_layers, neurons)
dbn, err_eqm = DBN.pretrain_DNN(dbn, n_epochs, lr, batch_size, X_train)

DBN_qualt_analysis(dbn, X_train, nb_images, nb_iterations, imagepath+"DBN_generated_char{0}".format(char) )