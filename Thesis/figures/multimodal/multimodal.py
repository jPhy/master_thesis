import numpy as np
from math import sin

class LogTarget(object):
    """Creates the log_target:
    log(sin(x[0]*.5)**10 * (1. + sin(x[1]*.5)**2) *
          * \prod_{i>=2} sin(x_i)**2 / x_i**2) inside [-6]*dim [+6]*dim,
    -inf outside.

        :param dim:

            integer; the dimension

    """
    def __init__(self, dim):
        assert dim >= 2
        self.dim = dim

    def __call__(self, x):
        if (x > -6.).all() and (x < 6.).all():
            out  = sin(x[0]*.5)**10
            out *= 1. + sin(x[1]*.5)**2
            for i in x[2:]:
               out *= min(1./4, 1./i**2) if i != 0. else 1.
            return np.log(out)
        else:
            return -np.inf
