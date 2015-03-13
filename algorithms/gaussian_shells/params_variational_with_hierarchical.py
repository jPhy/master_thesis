from __future__ import division
import numpy as np
import pypmc

params = dict(
random_seed = np.random.randint(2**62),
dim = 20,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 2*10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)

L = 2500, # patch length for mcmcmix (use this param to define the max number of components)

N_thin = 50, # use only every `N_thin`-th sample for the variational bayes (1 means all)

# how variational bayes is initialized, known keywords: ['package_default', 'means_only', 'initial_guess', 'initial_guess_large_nu'] (see actual program for details)
vb_initialization = 'initial_guess_large_nu',
N_max_vb = 100, # maximum number of steps in variational bayes
#vb_initial_K = 50,
vb_rel_tol = 10**-10,
vb_abs_tol = 10**-5,

kill_in_hc = True, # define if hierarchical clustering shall kill after each step or not (good in high, bad in low dim)
run_second_vb = False,

# how variational bayes is initialized, known keywords: ['means_only', 'initial_guess', !!!BUG!!! 'initial_guess_large_nu' !!!BUG!!!, 'initial_guess_large_nu_corrected'] (see actual program for details)
#vb2_initialization = 'initial_guess_large_nu_corrected',

abandon_weights = False, # delete the component weights obtained from the markov-chains (True proved to be worse in all experiments)
)
params.update((('vb_prune', params['L']/params['N_thin']),))
params.update((('pypmc_version', pypmc.__version__),))
