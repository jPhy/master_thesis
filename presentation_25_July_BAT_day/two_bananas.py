from pypmc.density.gauss import Gauss
import numpy as np
from math import exp

class SingleBanana(object):
    """Creates the log_target as in [WKB09]

        :param dim:

            integer; the dimension

    """
    def __init__(self, dim, twistdim):
        assert dim >= 2
        assert twistdim < dim - 1
        self.dim = dim
        self.twistdim = twistdim
        # kilbinger_params:
        self.sigma1_squared = 100.
        self.p = 10.
        self.b = .03
        self.mean = np.zeros(dim)
        self.cov = np.eye(dim)
        self.cov[twistdim, twistdim] = self.sigma1_squared
        self.underlying_gauss = Gauss(self.mean, self.cov)

    def __call__(self, x):
        x = np.array(x)
        x[1+self.twistdim] += self.b * (x[0+self.twistdim]*x[0+self.twistdim] - self.sigma1_squared)
        return self.underlying_gauss.evaluate(x)


class LogTarget(object):
    """Creates the log_target similar to that in [WKB09]:
    This target function will consist of two bananas in two different
    dimensions

        :param dim:

            integer; the dimension

    """
    def __init__(self, dim):
        assert dim >= 4
        self.dim = dim
        self.banana1 = SingleBanana(dim, 0)
        self.banana2 = SingleBanana(dim, 2)

    def __call__(self, x):
        x1 = np.array(x); x1[1] += 15; x1[3] -= 5
        x2 = np.array(x); x2[1] -=  5; x2[3] += 5
        return np.log( exp(self.banana1(x1)) + 30. * exp(self.banana2(x2)) )
