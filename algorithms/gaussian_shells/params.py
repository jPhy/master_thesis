import numpy as np

params = dict(
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

original_default2d = dict(
random_seed = np.random.randint(2**62),
dim = 2,
kill_in_hc = True,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)
K_g = 15, # components per group
L = 100,
critical_r = 2.,
)

original_default10d = dict(
random_seed = np.random.randint(2**62),
dim = 10,
kill_in_hc = True,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 2.* 10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)
K_g = 15, # components per group
L = 100,
critical_r = 2.,
)

original_default20d = dict(
random_seed = np.random.randint(2**62),
dim = 20,
kill_in_hc = True,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 2.* 10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)
K_g = 25, # components per group
L = 200,
critical_r = 2.,
)
