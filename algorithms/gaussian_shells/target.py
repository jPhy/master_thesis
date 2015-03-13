from math import sqrt, exp
import numpy as np

def gauss_shell(x, c, w=2., r=5.):
    """Evaluate the Gauss shell function with parameters
    ``c``, ``w`` and ``r`` at ``x``.

    The function is:
    1./np.sqrt(2.*np.pi*w**2) * np.exp(-(np.linalg.norm(x-c)-r)**2/(2.*w**2))

    """
    return 1./sqrt(2.*np.pi*w**2) * exp(-(np.linalg.norm(x-c)-r)**2/(2.*w**2))

class LogTarget(object):
    """Creates the log_target as in [Allen Fred paper]

        :param dim:

            integer; the dimension

    """
    def __init__(self, dim):
        self.dim = dim
        # define the target density
        # here, two Gaussian shells with parameters
        self.c_0 = np.array([+3.5] + (dim-1)*[0])
        self.c_1 = np.array([-3.5] + (dim-1)*[0])
        self.r = 2.0
        self.w = 0.1
        # and equal weight are used

    def __call__(self, x):
        return np.log(  .5 * (gauss_shell(x, self.c_0, self.w, self.r) + gauss_shell(x, self.c_1, self.w, self.r))  )
