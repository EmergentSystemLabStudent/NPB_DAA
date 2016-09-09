import numpy as np
from pyhsmm.basic.pybasicbayes.distributions import Categorical, Poisson
import matplotlib.pyplot as plt


def test_poisson():
    ret = np.zeros(100)
    for i in range(100):
        dist = Poisson(alpha_0 = 1, beta_0 = 0.1)
        ret[i] = dist.params['lmbda']

    plt.hist(ret)
    plt.show()


def test_categorical():
    ret = np.zeros((100, 10))
    for i in range(100):
        dist = Categorical(K = 10, alpha_0 = 1.0)
        ret[i, :] = dist.params['weights']

    plt.matshow(ret)
    plt.show()
