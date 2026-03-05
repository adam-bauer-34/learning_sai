import corner as cr
import numpy as np
import matplotlib.pyplot as plt

ndim, nsamples = 2, 10000

samp = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
figure = cr.corner(samp)

plt.show()
