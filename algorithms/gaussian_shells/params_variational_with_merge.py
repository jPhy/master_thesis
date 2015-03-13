from __future__ import division
import numpy as np
import pypmc

params = dict(
random_seed = np.random.randint(2**62),
dim = 2,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)

N_thin = 10, # use only every `N_thin`-th sample for the variational bayes (1 means all)

critical_r = 2., # the critical R-value for chain grouping
K_g = 15, # number of components per chain_group
L = 200, # patch length for short patches

vb_rel_tol = 10**-10,
vb_abs_tol = 10**-5,

vbm_prune = True, # prune components in VBMerge?

N_max_vb  = 1000, # maximum number of steps in first variational bayes
N_max_vbm = 5, # maximum number of steps in VBMerge

abandon_weights = False, # delete the component weights obtained from the markov-chains (True proved to be worse in all experiments)
)
params.update((('pypmc_version', pypmc.__version__),))
