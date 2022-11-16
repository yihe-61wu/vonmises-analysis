import numpy as np
from scipy.special import i0e, i1e, ive


def expectation_variance_vonmises(mu, kappa):
    """
    Find expectation and variance of Von Mises distribution in 2d plane.
    Reference: Hillen, et al. (2017) Mathematical Biosciences and Engineering
    Moments of von mises and fisher distributions and applications, equations (21) and (22)
    :param mu: mean angle in radius
    :param kappa: dispersion: when small, vonmises->normal; when larger, vonmises->uniform
    :return: expectation 2d vector and covariance 2x2 matrix
    """
    mu, kappa = np.atleast_1d(mu), np.atleast_1d(kappa)
    if mu.size == 1 and kappa.size > 1:
        mu = np.repeat(mu, kappa.size)
    elif mu.size > 1 and kappa.size == 1:
        kappa = np.repeat(kappa, mu.size)
    assert mu.size == kappa.size
    m_x, m_y = np.cos(mu), np.sin(mu)
    k_i0e, k_i1e, k_i2e = i0e(kappa), i1e(kappa), ive(2, kappa)
    r_i0i1, r_i0i2 = np.divide(k_i1e, k_i0e), np.divide(k_i2e, k_i0e)
    _E_xy = np.transpose([r_i0i1 * m_x, r_i0i1 * m_y])
    _Var_xy = np.einsum('k,ij->kij', 0.5 * (1 - r_i0i2), np.eye(2)) \
             + np.einsum('k,kij->kij', r_i0i2 - r_i0i1 ** 2, np.moveaxis([[m_x ** 2, m_x * m_y], [m_x * m_y, m_y ** 2]], 2, 0))
    E_xy, Var_xy = np.squeeze(_E_xy), np.squeeze(_Var_xy)
    return E_xy, Var_xy
