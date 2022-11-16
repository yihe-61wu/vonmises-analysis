import numpy as np
import numpy.random as nprd
from scipy.special import i0
import matplotlib.pyplot as plt


### example: comparison between gaussian and vonmises
mu, sigma = 0.0, 1.0
kappa = 1 / sigma ** 2  # makes vonmises similar to gaussian

# random sampling
s_gaussian = nprd.normal(mu, sigma, 1000)
s_vonmises = nprd.vonmises(mu, kappa, 1000)

# pdf
x = np.linspace(-np.pi, np.pi, num=361)
y_gaussian = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
y_vonmises = np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))

plt.figure()
for n, s, y, c in zip(('gaussian', 'vonmises'), (s_gaussian, s_vonmises), (y_gaussian, y_vonmises), ('tab:blue', 'tab:red')):
    plt.hist(s, 50, density=True, alpha=0.5, color=c)
    plt.plot(x, y, label=n, color=c)

plt.axvline(-np.pi, color='k', ls='-.')
plt.axvline(np.pi, color='k', ls='-.')
plt.title(r"$\mu=0.0$, $\sigma=1.0$, $\kappa = \sigma^{-2}$")
plt.legend()


### difference between gaussian and vonmises
gaussian = lambda sigma: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
vonmises = lambda sigma: np.exp(1 / sigma ** 2 * np.cos(x - mu)) / (2 * np.pi * i0(1 / sigma ** 2))

plt.figure()
for sigma in (0.05, 0.1, 0.5, 1):
    y = gaussian(sigma) - vonmises(sigma)
    plt.plot(x, y, label=r"$\sigma={}$".format(sigma))

plt.axvline(-np.pi, color='k', ls='-.')
plt.axvline(np.pi, color='k', ls='-.')
plt.title(r"pdf[gaussian] $-$ pdf[vonmises]")
plt.legend()

plt.show()
