#! /usr/bin/python
'outfilename is command line argument'

import sys, os
sys.path.insert( 0, os.path.abspath('..') )
outfilename = sys.argv[1]

import pypmc, eos, constraints as constr, numpy as np
from nuisance import get_nuisance
from pypmc.tools import plot_mixture
from pypmc.tools.convergence import perp, ess

# define log posterior function
constraints = constr.Bs_to_ll | constr.Bplus_to_Pll_low_recoil | constr.Bplus_to_Pll_large_recoil_sub_binned | constr.B_to_Pll_form_factors
nuisance = get_nuisance(constraints)
priors = [eos.LogPrior.Flat("Re{cS}", range_min=-15, range_max=15),
          eos.LogPrior.Flat("Im{cS}", range_min=-15, range_max=15)] + nuisance
dim = len(priors)
ind_lower = [p.range_min for p in priors]
ind_upper = [p.range_max for p in priors]
ind = pypmc.tools.indicator.hyperrectangle(ind_lower, ind_upper)
options = {"scan-mode": "cartesian", "model": "WilsonScan", "form-factors": "KMPW2010"}

ana = eos.Analysis(constraints, priors, options)

# merge with indicator
log_target = pypmc.tools.indicator.merge_function_with_indicator(ana, ind, -np.inf)


# define a proposal for the initial Markov chain run
mc_prop = pypmc.density.gauss.LocalGauss(np.diag([.01, .01] + [0.0001 * (p.upper - p.lower) for p in nuisance]))

# define initial points for the Markov chain run
def draw_uniform_in_support():
    sample = np.empty(dim)
    for d in range(dim):
        sample[d] = np.random.uniform(priors[d].range_min, priors[d].range_max)
    return sample
start = draw_uniform_in_support()
# log_target(start) must not be -inf!
while log_target(start) == -np.inf:
    start[:] = draw_uniform_in_support()

# define the Markov chain
mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, mc_prop, start, prealloc=100*1000)

# run chains and use self adaptation
for i in range(100):
    print 100 - i, 'more iteratios to be run'
    mc.run(1000)
    mc.adapt()


# save samples
np.save(outfilename, mc.history[:])
