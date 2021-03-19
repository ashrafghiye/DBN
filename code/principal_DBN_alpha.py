import utils
import principal_RBM_alpha as RBM
import numpy as np
import matplotlib.pyplot as plt


class DBN:

    def __init__(self, layers, hidden_units):
        """
        layers: minimum 1 layer -> RBM
        hidden_units: must be equal to layers + 1
        """

        assert (layers >= 1)
        self.layers = [None] * layers
        self.hidden_units = hidden_units
        self.num_layers = layers

        for layer in range(layers):
            self.layers[layer] = RBM.init_RBM(hidden_units[layer], hidden_units[layer + 1])


def init_DNN(layers, hidden_units):
  return DBN(layers, hidden_units)


def pretrain_DNN(dbn, epochs, lr, taille_batch, data):
    err_layers = []
    x = data.copy()

    for i in range(dbn.num_layers):
        dbn.layers[i], err_eqm = RBM.train_RBM(dbn.layers[i], epochs, lr, taille_batch, x)
        err_layers.append(err_eqm)
        x = RBM.entree_sortie_RBM(x, dbn.layers[i])

    return dbn, err_layers


def generer_image_DBN(dbn, nb_images, iter_gibbs):

  p, q = dbn.layers[0].a.shape[1], dbn.layers[-1].b.shape[1]
  imgs = []

  for i in range(0, nb_images):
    v = 1* np.random.rand(1,dbn.layers[-1].W.shape[0])<0.5

    for j in range(0, iter_gibbs):
      p_h = RBM.entree_sortie_RBM(v, dbn.layers[-1])
      h = 1* np.random.rand(p_h.shape[0],p_h.shape[1])<p_h
      p_v = RBM.sortie_entree_RBM(h, dbn.layers[-1])
      v = 1* np.random.rand(p_v.shape[0],p_v.shape[1])<p_v

    for l in range(dbn.num_layers-2, -1, -1):
      proba = RBM.sortie_entree_RBM(v, dbn.layers[l])
      v = 1* np.random.rand(proba.shape[0], proba.shape[1])<proba


    #fin generation
    imgs.append(1 * v.reshape(20, 16))
    plt.figure()
    plt.imshow(imgs[-1], cmap='gray') # AlphaDigits
    plt.title("Generated image after {0} iterations".format(iter_gibbs))
    plt.show()

  return np.array(imgs)

