"""
This script evaluate the performance of a RBM network
"""

import utils
import principal_RBM_alpha as RBM
import matplotlib.pyplot as plt
import numpy as np


imagepath = '../images/RBM_analysis/'


def RBM_quant_analysis(char='3', q = 240, lr = 0.2, n_epochs = 200, batch_size = 6, visualize=True, outputpath = imagepath):
  """
  Plot the reconstractive error after training with the specified hyperparamters for a given character
  """

  X_train = utils.lire_alpha_digit(char)
  p = X_train.shape[1]

  rbm = RBM.init_RBM(p, q)

  rbm, err_eqm = RBM.train_RBM(rbm, n_epochs, lr, batch_size, X_train)

  if visualize:
    plt.figure()
    plt.plot(range(n_epochs), err_eqm)
    plt.title('EQM: lr={0}, bs={1}, epochs={2}, q={3}, char={4}'.format(lr, batch_size, n_epochs, q, char))
    plt.xlabel('epochs')
    plt.ylabel('EQM')
    if imagepath.split('/')[-1] != "RBM_analysis":
      plt.savefig('{0}.png'.format(outputpath))
    plt.show()

  return err_eqm

def RBM_qualt_analysis(rbm, X_train, nb_images = 3, nb_iterations = 1000, outputpath = imagepath):
  """
  Plot some examples after training with the specified hyperparamters for a given character
  """
  x_generated_rbm = RBM.generer_image_RBM(rbm, nb_images, nb_iterations, visualize=False)
  utils.plot_examples_alphadigits(X_train, x_generated_rbm, nb_iterations, outputpath)



#initial parameters
char = 'C'
q = 160
lr = 0.2
n_epochs= 100
batch_size = 6

## Hyper-parameters range
lrs = [1e-2, 1e-1, 1.5e-1, 2e-1, 2.5e-1, 3e-1, 5e-1]
batch_sizes = [1, 3, 6, 9, 18, 39]
n_epochss = [20, 40, 80, 100, 200]
qs = [260, 240, 220, 180, 160, 120, 80]
chars = ['A', 'S', 'X', '0', '8', '3']

## Best Learning rate
errs_lr = []

for lr in lrs:
  errs_lr.append(RBM_quant_analysis(char, q, lr, n_epochs, batch_size, False)[-1])

plt.figure()
plt.plot(lrs, errs_lr, marker='o')
plt.title('EQM: bs={0}, epochs={1}, q={2}, char={3}'.format(batch_size, n_epochs, q, char))
plt.xlabel('learning rate')
plt.xticks(lrs)
plt.ylabel('EQM')
plt.show()

best_lr = lrs[np.argmin(errs_lr)]
best_lr

## Best latent dimension
errs_q = []

for q in qs:
  errs_q.append(RBM_quant_analysis(char, q, best_lr, n_epochs, batch_size, False)[-1])

plt.figure()
plt.plot(qs, errs_q, marker='o')
plt.title('EQM: bs={0}, epochs={1}, lr={2}, char={3}'.format(batch_size, n_epochs, best_lr, char))
plt.xlabel('q')
plt.xticks(qs)
plt.ylabel('EQM')
plt.show()

best_q = qs[np.argmin(errs_q)]
best_q

## Best batch size

errs_bs = []

for batch_size in batch_sizes:
  errs_bs.append(RBM_quant_analysis(char, best_q, best_lr, n_epochs, batch_size, False)[-1])

plt.figure()
plt.plot(batch_sizes, errs_bs, marker='o')
plt.title('EQM: q={0}, epochs={1}, lr={2}, char={3}'.format(best_q, n_epochs, best_lr, char))
plt.xlabel('batch size')
plt.xticks(batch_sizes)
plt.ylabel('EQM')
plt.show()

best_bs = batch_sizes[np.argmin(errs_bs)]
best_bs

## Learning curve
char = '3'
RBM_quant_analysis(char, best_q, best_lr, n_epochs, best_bs, True, imagepath+"RBM_eqm_char{0}".format(char));
char = 'C'
RBM_quant_analysis(char, best_q, best_lr, n_epochs, best_bs, True, imagepath+"RBM_eqm_char{0}".format(char));


## Qualitative analysis character 3

nb_images, nb_iterations = 3, 1000

char='3'
X_train = utils.lire_alpha_digit(char)
p = X_train.shape[1]

rbm = RBM.init_RBM(p, best_q)
rbm, err_eqm = RBM.train_RBM(rbm, n_epochs, best_lr, best_bs, X_train)

RBM_qualt_analysis(rbm, X_train, nb_images, nb_iterations, imagepath+"RBM_generated_char{0}".format(char))

char='C'
X_train = utils.lire_alpha_digit(char)
p = X_train.shape[1]

rbm = RBM.init_RBM(p, best_q)
rbm, err_eqm = RBM.train_RBM(rbm, n_epochs, best_lr, best_bs, X_train)

RBM_qualt_analysis(rbm, X_train, nb_images, nb_iterations, imagepath+"RBM_generated_char{0}".format(char))