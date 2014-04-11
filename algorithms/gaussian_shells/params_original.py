import numpy as np

params = dict(
random_seed = np.random.randint(2**62),
dim = 10,
kill_in_hc = True,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 2* 10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)
K_g = 15, # components per group
L = 100,
critical_r = 2.,
)


# ----------------------------------------
# do not change anything below this line !
# ----------------------------------------

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
N_MCMC = 2* 10**4, # number of steps per chain
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
N_MCMC = 2* 10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)
K_g = 25, # components per group
L = 200,
critical_r = 2.,
)
