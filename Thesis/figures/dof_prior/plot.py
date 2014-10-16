import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

function_evaluated = []
params = []

# The log of the dof conjugate prior
def lnT(tau, xi, sigma):
    x = 0.5 * tau
    return sigma * (x * np.log(x) - gammaln(x)) - xi * x

x_values = np.linspace(0.0001,16, 10**5)

def add_plot(xi, sigma, str_xi=None, str_sigma=None):
    if str_xi is None:
        str_xi = str(xi)
    else:
        assert float(str_xi) == xi
        str_xi = str_xi.replace(" ", r"\,")
    if str_sigma is None:
        str_sigma = str(sigma)
    else:
        assert float(str_sigma) == sigma
        str_sigma = str_sigma.replace(' ', r'\,')
    # evaluate lnT at the ``x_values`` with given parameter values
    y_values = np.asarray(map(lambda x: lnT(x, xi, sigma), x_values))
    # renormalize for numerical stability
    y_values -= y_values.max()
    y_values = np.exp(y_values)
    # calculate correctly normalized ``y_values``
    y_values /= np.trapz(y_values, x_values)
    # add to function_evaluated
    function_evaluated.append(y_values)
    params.append((str_xi, str_sigma))


# ------------------------------------------------------


# specify parameter values for the functions to be plot

add_plot(xi=  2, sigma=-0.9, str_xi='  2', str_sigma='-0.9')
add_plot(xi=  2, sigma=  0, str_xi='  2', str_sigma='  0')
add_plot(xi=  2, sigma=  1, str_xi='  2', str_sigma='  1')
add_plot(xi= 20, sigma= 10, str_xi=' 20', str_sigma=' 10')
add_plot(xi=200, sigma=100)
add_plot(xi=125, sigma=100)
add_plot(xi=110, sigma=100)


# -------------------------------------------------------


# plot specified functions
for i, y_values in enumerate(function_evaluated):
    str_xi = str(params[i][0])
    if (len(str_xi) < 3):
        str_xi = r"\," + str_xi if len(str_xi) == 1 else r"\,\," + str_xi
    str_sigma = str(params[i][1])
    if (len(str_sigma) < 3):
        str_sigma = r"\," if len(str_sigma) == 1 else r"\,\," + str_sigma
    label = r"$\xi=" + str_xi + r",\,\sigma=" + str_sigma + r"$"
    plt.plot(x_values, y_values, label=label)

plt.ylim(0, 3)
plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$\mathrm{T}\left(\tau \vert \xi, \sigma\right)$', fontsize=15)
plt.legend()
plt.savefig('plots_dof_prior.svg')
