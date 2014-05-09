from math import sin
import numpy as np

class LogTarget(object):
    """Creates the log_target:
    log(\prod_i sin(x_i)**2 / x_i**2) inside [-6]*dim [+6]*dim,
    -inf outside.

        :param dim:

            integer; the dimension

    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        if (x > -6.).all() and (x < 6.).all():
            out = 1.
            for i in x:
                out *= sin(i)**2 / i**2 if i != 0. else 1.
            return np.log(out)
        else:
            return -np.inf
