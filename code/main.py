from utils import load_mnist

import principal_DNN_MNIST as DNN
import principal_DBN_alpha as DBN

import matplotlib.pyplot as plt
import numpy as np

## initialization
X, X_test, y, y_test = load_mnist()
imagepath = '../images/DNN_analysis/'

## select a number in [1,2,3]
## 1 -> analyse as function of number of layers
## 2 -> analyse as function of number of hidden units
## 3 -> analyse as function of number of training examples

study_case = 1


### analysis as function of number of layers

def analyse_num_layers(X, X_test, y, y_test, imagepath):
    ##fixed parameteres

    p = X.shape[1]
    q = 10

    hidden_units = 200
    n_epochs_rbm = 30
    n_epochs_retro = 20
    lr = 0.15
    batch_size = 100

    training_samples = X.shape[0] // 2
    X_train, y_train = X[:training_samples, :], y[:training_samples]

    ##variable parameteres

    test_layers = 5

    first_layer = [(p, hidden_units)]
    last_layer = [(hidden_units, q)]
    neurons_v = [[(p, hidden_units), (hidden_units, q)]]

    for i in range(1, test_layers):
        neurons_v += [first_layer + [(hidden_units, hidden_units)] * i + last_layer]

    ##track of error the tests
    acc_pre, crossentropy_pre = [], []
    acc_norm, crossentropy_norm = [], []

    for neurons in neurons_v:
        num_layers = len(neurons)

        # initialize identical networks
        dnn_pre = DBN.init_DNN(num_layers, neurons)
        dnn_norm = DNN.copy_dnn(dnn_pre)

        # pre-entrain one and fine-tune
        dnn_pre, _ = DBN.pretrain_DNN(dnn_pre, n_epochs_rbm, lr, batch_size, X_train)
        dnn_pre, _ = DNN.retropropagation(dnn_pre, X_train, y_train, n_epochs_retro, lr, batch_size, "pre-trained", False)
        print("Pre-trained with {0} hidden layers".format(num_layers - 1))
        test_scores = DNN.test_DNN(dnn_pre, X_test, y_test, "", False)
        acc_pre.append(test_scores[0])
        crossentropy_pre.append(test_scores[1])

        print("\n")

        # use the initial network without pre-entraining
        dnn_norm, _ = DNN.retropropagation(dnn_norm, X_train, y_train, n_epochs_retro, lr, batch_size, "normal", False)
        print("Normal with {0} hidden layers".format(num_layers - 1))
        test_scores = DNN.test_DNN(dnn_norm, X_test, y_test, "", False)
        acc_norm.append(test_scores[0])
        crossentropy_norm.append(test_scores[1])

        print("-----------------")

        ##visualization
        variables = np.arange(1, test_layers + 1)

        plt.figure()
        plt.plot(variables, acc_pre, marker='^', label='pre-trained')
        plt.plot(variables, acc_norm, marker='o', label='normal')
        plt.xticks(variables)
        plt.xlabel('num of layers')
        plt.ylabel('accuracy')
        plt.title('accuracy vs num of layers')
        plt.legend()
        plt.savefig(imagepath + "acc_vs_numlayers.png")
        plt.show()

        ##visualization
        variables = np.arange(1, test_layers + 1)

        plt.figure()
        plt.plot(variables, crossentropy_pre, marker='^', label='pre-trained')
        plt.plot(variables, crossentropy_norm, marker='o', label='normal')
        plt.xticks(variables)
        plt.xlabel('num of layers')
        plt.ylabel('cross-entropy')
        plt.title('cross-entropy vs num of layers')
        plt.legend()
        plt.savefig(imagepath + "crosse_vs_numlayers.png")
        plt.show()


def analyse_hidden_units(X, X_test, y, y_test, imagepath):
    ##fixed parameteres

    p = X.shape[1]
    q = 10

    num_layers = 3  # 2 hidden layers
    n_epochs_rbm = 30
    n_epochs_retro = 20
    lr = 0.15
    batch_size = 100

    training_samples = X.shape[0] // 2
    X_train, y_train = X[:training_samples, :], y[:training_samples]

    ##variable parameteres

    hidden_units_v = [100, 200, 300, 500, 1000]
    neurons_v = []
    for hidden_units in hidden_units_v:
        neurons_v.append([(p, hidden_units), (hidden_units, hidden_units), (hidden_units, q)])

    ##track of error the tests
    acc2_pre, crossentropy2_pre = [], []
    acc2_norm, crossentropy2_norm = [], []
    for idx, neurons in enumerate(neurons_v):
        # initialize identical networks
        dnn_pre = DBN.init_DNN(num_layers, neurons)
        dnn_norm = DNN.copy_dnn(dnn_pre)

        # pre-entrain one and fine-tune
        dnn_pre, _ = DBN.pretrain_DNN(dnn_pre, n_epochs_rbm, lr, batch_size, X_train)
        dnn_pre, _ = DNN.retropropagation(dnn_pre, X_train, y_train, n_epochs_retro, lr, batch_size, "pre-trained", False)
        print("Pre-trained with {0} hidden units".format(hidden_units_v[idx]))
        test_scores = DNN.test_DNN(dnn_pre, X_test, y_test, "", False)
        acc2_pre.append(test_scores[0])
        crossentropy2_pre.append(test_scores[1])

        print("\n")

        # use the initial network without pre-entraining
        dnn_norm, _ = DNN.retropropagation(dnn_norm, X_train, y_train, n_epochs_retro, lr, batch_size, "normal", False)
        print("Normal with {0} hidden units".format(hidden_units_v[idx]))
        test_scores = DNN.test_DNN(dnn_norm, X_test, y_test, "", False)
        acc2_norm.append(test_scores[0])
        crossentropy2_norm.append(test_scores[1])

        print("-----------------")

        ##visualization
        variables = np.arange(1, len(hidden_units_v) + 1)

        plt.figure()
        plt.plot(variables, acc2_pre, marker='^', label='pre-trained')
        plt.plot(variables, acc2_norm, marker='o', label='normal')
        plt.xticks(variables, hidden_units_v)
        plt.xlabel('hidden units')
        plt.ylabel('accuracy')
        plt.title('accuracy vs hidden units')
        plt.legend()
        plt.savefig(imagepath + "acc_vs_hiddenunits.png")
        plt.show()

        ##visualization
        variables = np.arange(1, len(hidden_units_v) + 1)

        plt.figure()
        plt.plot(variables, crossentropy2_pre, marker='^', label='pre-trained')
        plt.plot(variables, crossentropy2_norm, marker='o', label='normal')
        plt.xticks(variables, hidden_units_v)
        plt.xlabel('hidden units')
        plt.ylabel('cross-entropy')
        plt.title('cross-entropy vs hidden units')
        plt.legend()
        plt.savefig(imagepath + "crosse_vs_hiddenunits.png")
        plt.show()

def analyse_training_samples(X, X_test, y, y_test, imagepath):
    ##fixed parameteres
    p = X.shape[1]
    q = 10

    hidden_units = 200
    num_layers = 3  # 2 hidden layers
    neurons = [(p, hidden_units), (hidden_units, hidden_units), (hidden_units, q)]
    n_epochs_rbm = 20
    n_epochs_retro = 20
    lr = 0.15
    batch_size = 100

    ##variable parameteres
    training_samples_v = [1000, 3000, 7000, 10000, 30000, 60000]

    ##track of error the tests
    acc3_pre, crossentropy3_pre = [], []
    acc3_norm, crossentropy3_norm = [], []

    for training_samples in training_samples_v:
        X_train, y_train = X[:training_samples, :], y[:training_samples]

        # initialize identical networks
        dnn_pre = DBN.init_DNN(num_layers, neurons)
        dnn_norm = DNN.copy_dnn(dnn_pre)

        # pre-entrain one and fine-tune
        dnn_pre, _ = DBN.pretrain_DNN(dnn_pre, n_epochs_rbm, lr, batch_size, X_train)
        dnn_pre, _ = DNN.retropropagation(dnn_pre, X_train, y_train, n_epochs_retro, lr, batch_size, "pre-trained", False)
        print("Pre-trained with {0} training samples".format(training_samples))
        test_scores = DNN.test_DNN(dnn_pre, X_test, y_test, "", False)
        acc3_pre.append(test_scores[0])
        crossentropy3_pre.append(test_scores[1])

        print("\n")

        # use the initial network without pre-entraining
        dnn_norm, _ = DNN.retropropagation(dnn_norm, X_train, y_train, n_epochs_retro, lr, batch_size, "normal", False)
        print("Normal with {0} training samples".format(training_samples))
        test_scores = DNN.test_DNN(dnn_norm, X_test, y_test, "", False)
        acc3_norm.append(test_scores[0])
        crossentropy3_norm.append(test_scores[1])

        print("-----------------")

        ##visualization
        variables = np.arange(1, len(training_samples_v) + 1)

        plt.figure()
        plt.plot(variables, acc3_pre, marker='^', label='pre-trained')
        plt.plot(variables, acc3_norm, marker='o', label='normal')
        plt.xticks(variables, training_samples_v)
        plt.xlabel('training samples')
        plt.ylabel('accuracy')
        plt.title('accuracy vs training samples')
        plt.legend()
        plt.savefig(imagepath + "acc_vs_trsamples.png")
        plt.show()

        ##visualization
        variables = np.arange(1, len(training_samples_v) + 1)

        plt.figure()
        plt.plot(variables, crossentropy3_pre, marker='^', label='pre-trained')
        plt.plot(variables, crossentropy3_norm, marker='o', label='normal')
        plt.xticks(variables, training_samples_v)
        plt.xlabel('training samples')
        plt.ylabel('cross-entropy')
        plt.title('cross-entropy vs training samples')
        plt.legend()
        plt.savefig(imagepath + "crosse_vs_trsamples.png")
        plt.show()


if study_case == 1:
    analyse_num_layers(X, X_test, y, y_test, imagepath)
elif study_case == 2:
    analyse_hidden_units(X, X_test, y, y_test, imagepath)
else:
    analyse_training_samples(X, X_test, y, y_test, imagepath)