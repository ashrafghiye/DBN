import utils
import principal_RBM_alpha as RBM
import matplotlib.pyplot as plt


X_train = utils.lire_alpha_digit('3')

p = X_train.shape[1]
q = 240

rbm = RBM.init_RBM(p, q)

n_epochs= 200
lr = 0.2
batch_size = 6

rbm, err_eqm = RBM.train_RBM(rbm, n_epochs, lr, batch_size, X_train)

plt.figure()
plt.plot(range(n_epochs), err_eqm)
plt.title('EQM vs epochs')
plt.show()

x_generated_rbm = RBM.generer_image_RBM(rbm, 3, 20);

utils.plot_examples_alphadigits(X_train, x_generated_rbm)