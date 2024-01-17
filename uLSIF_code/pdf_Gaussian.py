import numpy as np

def pdf_Gaussian(x, mu, sigma):
    d, nx = x.shape

    tmp = (x - np.tile(mu, (nx, 1)).T) / np.tile(sigma, (nx, 1)).T / np.sqrt(2)
    px = (2 * np.pi)**(-d / 2) / np.prod(sigma) * np.exp(-np.sum(tmp**2, axis=0))

    return px