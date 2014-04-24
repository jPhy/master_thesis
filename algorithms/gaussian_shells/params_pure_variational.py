import numpy as np
import pypmc

params = dict(
random_seed = np.random.randint(2**62),
dim = 20,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 2*10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)

N_thin = 10, # use only every `N_thin`-th sample for the variational bayes (1 means all)

# how variational bayes is initialized, known keywords: ['package_default', 'initial_guess_large_nu', 'means_only', 'long_patches', 'long_patches_rescaled_cov'] (see actual program for details)
vb_initialization = 'long_patches',
# cov_rescale_factor = 2., # the factor multiplied to the empirical long-patch covariances (only needed if vb_initialization = 'long_patches_rescaled_cov')
critical_r = 2., # the critical R-value for chain grouping (only needed if vb_initialization = 'long_patches')
K_g = 25, # number of components per chain_group (only neede for vb_initialization 'long_patches')
#L = 500, # patch length; only needed for vb_initialization 'initial_guess_large_nu' and 'means_only'
N_max_vb = 1000, # maximum number of steps in variational bayes
#vb_initial_K = 2,
vb_rel_tol = 10**-10,
vb_abs_tol = 10**-5,
abandon_weights = False,
)
# params.update((('vb_prune', params['L']/params['N_thin']),)) # prune components with N_k less than ``vb_prune``
params.update((('pypmc_version', pypmc.__version__),))


# ----------------------------------------
# do not change anything below this line !
# ----------------------------------------

pure_variational_default = dict(
random_seed = np.random.randint(2**62),
dim = 2,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)

N_thin = 100, # use only every `N_thin`-th sample for the variational bayes (1 means all)

# how variational bayes is initialized, known keywords: ['package_default'] (see actual program for details)
vb_initialization = 'package_default',
N_max_vb = 100, # maximum number of steps in variational bayes
vb_initial_K = 100,
vb_rel_tol = 10**-10,
vb_abs_tol = 10**-5,
)

pure_variational_good = dict(
random_seed = np.random.randint(2**62),
dim = 10,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 2*10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)

N_thin = 100, # use only every `N_thin`-th sample for the variational bayes (1 means all)

# how variational bayes is initialized, known keywords: ['package_default', 'initial_guess_large_nu', 'means_only'] (see actual program for details)
vb_initialization = 'initial_guess_large_nu',
L = 100, # patch length; only needed for vb_initialization 'initial_guess_large_nu' and 'means_only'
N_max_vb = 100, # maximum number of steps in variational bayes
#vb_initial_K = 2,
#vb_prune = 20, # prune components with N_k less than ``vb_prune``
vb_rel_tol = 10**-10,
vb_abs_tol = 10**-5,
abandon_weights = False,
)
