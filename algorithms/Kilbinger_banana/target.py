from pypmc.density.gauss import Gauss
import numpy as np

class LogTarget(object):
    """Creates the log_target as in [WKB09]

        :param dim:

            integer; the dimension

    """
    def __init__(self, dim):
        assert dim >= 2
        self.dim = dim
        # kilbinger_params:
        self.sigma1_squared = 100.
        self.p = 10.
        self.b = .03
        self.mean = np.zeros(dim)
        self.cov = np.diag([self.sigma1_squared] + [1. for i in range(dim-1)])
        self.underlying_gauss = Gauss(self.mean, self.cov)

    def __call__(self, x):
        x = np.array(x)
        x[1] += self.b * (x[0]*x[0] - self.sigma1_squared)
        return self.underlying_gauss.evaluate(x)
