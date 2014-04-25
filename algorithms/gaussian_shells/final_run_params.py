import numpy as np
import pypmc

params = dict(
random_seed = np.random.randint(2**62),
dim = 2,
method = 'original', # the algorithm to be executed, known are ['original', 'PMC_cornuet', 'VB_pripos', 'VB_cornuet']
first_proposal_index = 'average_original', # the proposal to be taken from database, can be int or ['good, average' _ 'original, variational']
max_sampling_steps = 20, # the maximum number of sampling runs
N_c = 200, # number of samples per component
#max_vb_steps = 100, # the maximum number of steps VB is allowed to run in each sampling (only if method = 'VB_...')
#first_VB_prior = False, # use component means and covs in ``first_proposal`` as prior (only if method = 'VB_...' and first_proposal_index = '..._variational')
rb = True, # use rao-blackwellized PMC? (only if method = 'original' or 'PMC_cornuet')
mincount = 20, # the minimum number of samples a component must have produced
)
params.update((('pypmc_version', pypmc.__version__),))


# ----------------------------------------
# do not change anything below this line !
# ----------------------------------------

good10d = dict(
random_seed = np.random.randint(2**62),
dim = 10,
method = 'VB_pripos', # the algorithm to be executed, known are ['original', 'PMC_cornuet', 'VB_pripos', 'VB_cornuet']
first_proposal_index = 499, # the proposal to be taken from database, can be int or ['good, average' _ 'original, variational']
max_sampling_steps = 20, # the maximum number of sampling runs
N_c = 400, # number of samples per component
max_vb_steps = 100, # the maximum number of steps VB is allowed to run in each sampling (only if method = 'VB_...')
first_VB_prior = True, # use component means and covs in ``first_proposal`` as prior (only if method = 'VB_...' and first_proposal_index = '..._variational')
#rb = True, # use rao-blackwellized PMC? (only if method = 'original' or 'PMC_cornuet')
mincount = 20, # the minimum number of samples a component must have produced
)

default2d = dict(
random_seed = np.random.randint(2**62),
dim = 2,
method = 'original', # the algorithm to be executed, known are ['original', 'PMC_cornuet', 'VB_pripos', 'VB_cornuet']
first_proposal_index = 'average_original', # the proposal to be taken from database, can be int or ['good, average' _ 'original, variational']
max_sampling_steps = 20, # the maximum number of sampling runs
N_c = 200, # number of samples per component
#max_vb_steps = 100, # the maximum number of steps VB is allowed to run in each sampling (only if method = 'VB_...')
#first_VB_prior = False, # use component means and covs in ``first_proposal`` as prior (only if method = 'VB_...' and first_proposal_index = '..._variational')
rb = True, # use rao-blackwellized PMC? (only if method = 'original' or 'PMC_cornuet')
mincount = 20, # the minimum number of samples a component must have produced
)

default10d = dict(
random_seed = np.random.randint(2**62),
dim = 10,
method = 'original', # the algorithm to be executed, known are ['original', 'PMC_cornuet', 'VB_pripos', 'VB_cornuet']
first_proposal_index = 'average_original', # the proposal to be taken from database, can be int or ['good, average' _ 'original, variational']
max_sampling_steps = 20, # the maximum number of sampling runs
N_c = 400, # number of samples per component
#max_vb_steps = 100, # the maximum number of steps VB is allowed to run in each sampling (only if method = 'VB_...')
#first_VB_prior = False, # use component means and covs in ``first_proposal`` as prior (only if method = 'VB_...' and first_proposal_index = '..._variational')
rb = True, # use rao-blackwellized PMC? (only if method = 'original' or 'PMC_cornuet')
mincount = 20, # the minimum number of samples a component must have produced
)

default20d = dict(
random_seed = np.random.randint(2**62),
dim = 20,
method = 'original', # the algorithm to be executed, known are ['original', 'PMC_cornuet', 'VB_pripos', 'VB_cornuet']
first_proposal_index = 'average_original', # the proposal to be taken from database, can be int or ['good, average' _ 'original, variational']
max_sampling_steps = 20, # the maximum number of sampling runs
N_c = 600, # number of samples per component
#max_vb_steps = 100, # the maximum number of steps VB is allowed to run in each sampling (only if method = 'VB_...')
#first_VB_prior = False, # use component means and covs in ``first_proposal`` as prior (only if method = 'VB_...' and first_proposal_index = '..._variational')
rb = True, # use rao-blackwellized PMC? (only if method = 'original' or 'PMC_cornuet')
mincount = 20, # the minimum number of samples a component must have produced
)
