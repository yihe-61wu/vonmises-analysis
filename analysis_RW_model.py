import numpy as np
import numpy.random as nprd
from scipy.special import i0e, i1e, ive
import matplotlib.pyplot as plt
from utils import expectation_variance_vonmises


### without loss of generosity, the mind wants to move to right, with step size 1

# example position
plt.figure()
n_sample = 666
for kappa, y_offset, color in zip((0.1, 1, 10), (0.1, 0, -0.1), ('tab:blue', 'tab:orange', 'tab:green')):
    s_rad = nprd.vonmises(0, kappa, n_sample)
    x_s, y_s = np.cos(s_rad), np.sin(s_rad) + y_offset
    avg = np.mean([x_s, y_s], axis=1)

    plt.axhline(y_offset, c=color, ls='dashed')
    plt.scatter(*avg, label=r"$\kappa={}$".format(kappa), marker='x', s=100, c=color)
    plt.scatter(x_s, y_s, alpha=0.5, s=1, c=color)

plt.xlabel("x")
plt.ylabel("y")
plt.title("example and average positions")
plt.legend(loc=3)
plt.grid()

# mean error
n_pts = 1001
n_halfpts = n_pts // 2
kappa = 10 ** np.flip(np.linspace(-7, 7, n_pts))
V_mean, V_var = expectation_variance_vonmises(0, kappa)

dV_mean = np.linalg.norm(V_mean - np.array([1, 0]), axis=1)
dV_var = 2 - dV_mean ** 2 - V_mean[:, 0] * 2
dV_std = np.sqrt(dV_var)

sigma2 = 1 / kappa
dG_mean = np.zeros_like(sigma2)
G_yvar = sigma2
dG_std = np.sqrt(G_yvar)

plt.figure()
plt.plot(kappa, 1 - V_mean[:, 0], label="x-mean")
plt.plot(kappa, 0 - V_mean[:, 1], label="y-mean")
plt.plot(kappa, np.sqrt(V_var[:, 0, 0]), label="x-std", ls='dotted')
plt.plot(kappa, np.sqrt(V_var[:, 1, 1]), label="y-std", ls='dotted')

plt.title(r"mind-body positional difference")
plt.xlabel(r"$\kappa$")
plt.ylabel("distance")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()

plt.figure()
plt.plot(kappa, dV_mean, label="vonmises-mean")
plt.plot(kappa, dV_std, label="vonmises-std")
plt.plot(kappa, dG_mean, label="gaussian-mean: 0", ls='-.')
plt.plot(kappa, dG_std, label="gaussian-std", ls='-.')
plt.plot(kappa, 0.5/kappa, label=r"$\kappa^{-1}/2$", ls='-.')

plt.xlabel(r"$\kappa$")
plt.ylabel("distance")
plt.title(r"mind-body positional difference")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()

plt.figure()
plt.plot(kappa[:n_halfpts], np.abs(dV_mean - dG_mean)[:n_halfpts], label="vonmises-gaussian mean")
plt.plot(kappa[:n_halfpts], np.abs(dV_std - dG_std)[:n_halfpts], label="vonmises-gaussian std")
for kc in (0.5, 0.4, 0.6, 1.0):
    label = r"vonmises mean-{}".format(kc) + "$\kappa^{-1}$"
    plt.plot(kappa[:n_halfpts], np.abs(dV_mean - kc/kappa)[:n_halfpts], label=label, ls='dotted')
    # not understand why 0.5 gives the best approximation
plt.title(r"difference between VonMises and Gaussian, $\kappa=\sigma^{-2}$")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()

### Remarks:
### the above two figures demonstrate the importance of decent motor control
### positional error will be driven by angular control when its error is large
### below I consider only kappa >= 1
### when kappa = 1, mean positional error > 1
### in addition, for gaussian, consider sigma <= 1
### when sigma = 1, mean = sigma * sqrt(2 / pi) = 0.7979...



# sigma = 0.0
#
# # random sampling
# # s_gaussian = np.random.normal(mu, sigma, 1000)
# rotation_mind = 0
# rotation_body = nprd.vonmises(rotation_mind, kappa, n_sample)
#
# x_body = np.cos(rotation_body) + nprd.normal(scale=sigma, size=n_sample)
# y_body = np.sin(rotation_body) + nprd.normal(scale=sigma, size=n_sample)
#
# dis_mind_body = np.linalg.norm(np.array([x_body - 1, y_body]), axis=0)
#
#
#
# plt.figure()
# plt.scatter(x_body, y_body)
#
# plt.figure()
# plt.hist(dis_mind_body, 100, density=True)
# # # print(rotation_body)
# # # print(x_body)
# #
#


plt.show()
