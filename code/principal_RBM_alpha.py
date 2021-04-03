import utils
import numpy as np
import matplotlib.pyplot as plt


class RBM:
  def __init__(self, p, q, sigma):
    """
    sigma: ecart-type des poids
    """
    self.W = np.random.normal(0, sigma, size=(p,q))
    self.a = np.zeros(shape=(1, p))
    self.b = np.zeros(shape=(1, q))

def init_RBM(p, q, sigma=0.01):
  return RBM(p,q,sigma)


def entree_sortie_RBM(data, rbm):
  """
  data: matrice de taille mxp
  rbm: object du type RBM.
  """

  logits = data @ rbm.W + rbm.b

  return utils.sigmoid(logits)


def sortie_entree_RBM(data, rbm):
  """
  data: matrice de taille mxq
  rbm: object du type RBM.
  """

  logits = data @ rbm.W.T + rbm.a

  return utils.sigmoid(logits)


def train_RBM(rbm, epochs, lr, taille_batch, data):
    """
    faire le training du RBM en utilisant l'algorithm CD-1
    """

    n = data.shape[0]
    p, q = rbm.a.shape[1], rbm.b.shape[1]
    shuffled_index = np.arange(n)
    err_eqm = []

    for i in range(0, epochs):
        np.random.shuffle(shuffled_index)
        x = data[shuffled_index]
        for batch in range(0, n, taille_batch):
            data_batch = x[batch:min(batch + taille_batch, n), :]
            taille_batch = data_batch.shape[0]

            v0 = data_batch
            p_h_v0 = entree_sortie_RBM(v0, rbm)
            h_0 = 1 * (np.random.rand(taille_batch, q) < p_h_v0)

            p_v_h0 = sortie_entree_RBM(h_0, rbm)
            v1 = 1 * (np.random.rand(taille_batch, p) < p_v_h0)
            p_h_v1 = entree_sortie_RBM(v1, rbm)

            da = np.sum(v0 - v1, axis=0)
            db = np.sum(p_h_v0 - p_h_v1, axis=0)
            dW = v0.T @ p_h_v0 - v1.T @ p_h_v1

            rbm.W += lr * dW / taille_batch
            rbm.a += lr * da / taille_batch
            rbm.b += lr * db / taille_batch

            # fin du batch
        h = entree_sortie_RBM(data, rbm)
        x_recovered = sortie_entree_RBM(h, rbm)
        err = np.mean(np.sum((data - x_recovered) ** 2, axis=1))
        err_eqm.append(err)

    return rbm, err_eqm

def generer_image_RBM(rbm, nb_images, iter_gibbs, visualize = True):

  p, q = rbm.a.shape[1], rbm.b.shape[1]
  imgs = []
  for i in range(0, nb_images):
    v = 1* np.random.rand(1,p)<0.5
    for j in range(0, iter_gibbs):
      p_h = entree_sortie_RBM(v, rbm)
      h = 1* np.random.rand(1,q)<p_h
      p_v = sortie_entree_RBM(h, rbm)
      v = 1* np.random.rand(1,p)<p_v

    #fin generation
    imgs.append(1 * v.reshape(20, 16))
    if visualize:
        plt.figure()
        plt.imshow(imgs[-1], cmap='gray') # AlphaDigits
        plt.title("Generated image after {0} iterations".format(iter_gibbs))
        plt.show()

  return np.array(imgs)

