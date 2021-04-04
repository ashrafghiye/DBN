from utils import load_mnist

import principal_DNN_MNIST as DNN
import principal_DBN_alpha as DBN

X_train, X_test, y_train, y_test = load_mnist()

p = X_train.shape[1]
q = 10

neurons = [(p, 300), (300, 300), (300, 300), (300,q)]
num_layers = len(neurons)
n_epochs_rbm = 60
n_epochs_retro = 100
lr = 0.2
batch_size = 100

dnn = DBN.init_DNN(num_layers, neurons)
dnn_normal = DNN.copy_dnn(dnn)

dnn, _ = DBN.pretrain_DNN(dnn, n_epochs_rbm, lr, batch_size, X_train)

dnn = DNN.retropropagation(dnn, X_train, y_train, n_epochs_retro, lr, batch_size, "pre-trained")

DNN.test_DNN(dnn, X_test, y_test, "Pre-trained", True, "../images/DNN_analysis/pretrain_final.png")

dnn_normal = DNN.retropropagation(dnn_normal, X_train, y_train, n_epochs_retro, lr, batch_size, "normal")

DNN.test_DNN(dnn_normal, X_test, y_test, "normal", True, "../images/DNN_analysis/normal_final.png")