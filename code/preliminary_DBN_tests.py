import utils
import principal_DBN_alpha as DBN
import matplotlib.pyplot as plt


X_train = utils.lire_alpha_digit('3')

p = X_train.shape[1]
q = 240
num_layers = 3

dbn = DBN.init_DNN(num_layers, [p, p//2, p//4, p//6])

n_epochs= 200
lr = 0.2
batch_size = 6

dbn, err_eqm = DBN.pretrain_DNN(dbn, n_epochs, lr, batch_size, X_train)

plt.figure()
layer = 2
plt.plot(range(n_epochs), err_eqm[layer])
plt.title('EQM vs epochs layer {0}'.format(layer))
plt.show()

x_generated_dbn = DBN.generer_image_DBN(dbn, 3, 25);

utils.plot_examples_alphadigits(X_train, x_generated_dbn)