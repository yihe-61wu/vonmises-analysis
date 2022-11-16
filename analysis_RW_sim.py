import numpy as np
import numpy.random as nprd
import matplotlib.pyplot as plt


kappa = 1e6
trial = 1000
duration = 1000
plot_example = 1

rot = nprd.vonmises(0, kappa, (trial, duration))
orn = np.cumsum(rot, axis=1)
step = np.swapaxes([np.cos(orn), np.sin(orn)], 0, 1)
pos = np.cumsum(step, axis=2)
pos_swap = np.swapaxes(pos, 0, 2)
print(pos.shape)
print(pos_swap.shape)

pos_mean = np.mean(pos_swap, axis=2)
pos_cov = np.array([np.cov(xy) for xy in pos_swap])

target = np.repeat([np.vstack((np.arange(1, duration + 1), np.zeros(duration)))], trial, axis=0)
tar_swap = np.swapaxes(target, 0, 2)
dist = np.linalg.norm(pos_swap - tar_swap, axis=1)
dist_mean = np.mean(dist, axis=1)
dist_var = np.var(dist, axis=1)

plt.figure('distance')
plt.plot(dist_mean, label='mean')
plt.plot(np.sqrt(dist_var), label='s.d.')
plt.grid()
plt.legend()
# plt.yscale('log')

plt.figure('mean position')
plt.plot(*pos_mean.T)
plt.grid()

plt.figure('covariance position')
plt.plot(pos_cov[:, 0, 0], label='x')
plt.plot(pos_cov[:, 1, 0], label='xy')
plt.plot(pos_cov[:, 1, 1], label='y')
plt.legend()
plt.grid()
plt.yscale('log')

# to add mean distance

if plot_example:
    plt.figure('rotation')
    plt.plot(rot[:10].T)
    plt.grid()

    plt.figure('orientation')
    plt.plot(orn[:10].T)
    plt.grid()

    plt.figure('step')
    [plt.scatter(*xy, alpha=0.1, marker='.') for xy in step[:10]]
    plt.grid()

    plt.figure('position')
    [plt.plot(*xy) for xy in pos[:10]]
    plt.grid()

    plt.figure('position2')
    [plt.scatter(*xy, alpha=0.5) for xy in pos_swap[:, :, :10]]
    plt.grid()

plt.show()
