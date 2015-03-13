import numpy as np
import pypmc
from mcmc import create_mcmc_samples
from long_patches import make_long_patch_gaussian_mixture

def make_first_proposal(log_target,
    # MCMC parameters
    k=10, N_burn_in=10**3, N_MCMC=2*10**4, N_adapt=500, scale=.1, upper=3., lower=-3.,
    # long_patch parameter
    K_g=15, critical_r=2.,
    N_thin=10):

    '''Run a black box algorithm to obtain a gaussian mixture suitable
    to importance sample the target. Also return samples obtained
    by multiple (``k``) Markov-chains and the `GaussianInference` object.
    ``log_target`` is the log of the target function. For additional
    parameters refer to the functions ``create_mcmc_samples``,
    ``make_long_patch_gaussian_mixture`` and the code in this function.
    Dimension is read as ``log_target.dim``

    '''
    mcmc_data = create_mcmc_samples(log_target, k, N_burn_in, N_MCMC, N_adapt, scale, upper, lower)
    mcmcmix = make_long_patch_gaussian_mixture(mcmc_data, K_g, critical_r)
    vb = pypmc.mix_adapt.variational.GaussianInference(np.vstack(mcmc_data)[::N_thin], initial_guess=mcmcmix)
    vb.run(1000, prune=.5*len(vb.data)/len(mcmcmix), rel_tol=1e-10, abs_tol=1e-5, verbose=True)
    return vb.make_mixture(), mcmc_data, vb
