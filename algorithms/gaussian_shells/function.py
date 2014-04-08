from math import sqrt, exp
import numpy as np

def gauss_shell(x, c, w=2., r=5.):
    """Evaluate the Gauss shell function with parameters
    ``c``, ``w`` and ``r`` at ``x``.

    The function is:
    1./np.sqrt(2.*np.pi*w**2) * np.exp(-(np.linalg.norm(x-c)-r)**2/(2.*w**2))

    """
    return 1./sqrt(2.*np.pi*w**2) * exp(-(np.linalg.norm(x-c)-r)**2/(2.*w**2))
