import utils



X_train, _, y_train, _ = utils.load_mnist()

utils.visualize_mnist_examples(X_train, y_train)
utils.visualize_alphadigits_examples()