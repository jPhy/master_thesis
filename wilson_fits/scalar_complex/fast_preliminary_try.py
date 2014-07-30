import sys, os
sys.path.insert( 0, os.path.abspath('..') )

import pypmc, eos, constraints as constr, numpy as np
from nuisance import get_nuisance
from pypmc.tools import plot_mixture
from pypmc.tools.convergence import perp, ess

# define log posterior function
constraints = constr.Bs_to_ll
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


# Markov chain prerun --> run 10 Markov chains with different initial points

# define a proposal for the initial Markov chain run
mc_prop = pypmc.density.gauss.LocalGauss(np.diag([.01, .01] + [0.0001 * (p.upper - p.lower) for p in nuisance]))

# define initial points for the Markov chain run
def draw_uniform_in_support():
    sample = np.empty(dim)
    for d in range(dim):
        sample[d] = np.random.uniform(priors[d].range_min, priors[d].range_max)
    return sample
starts = np.array([draw_uniform_in_support() for i in range(10)])
# log_target(starts[i]) must not be -inf!
for start in starts:
    while log_target(start) == -np.inf:
        start[:] = draw_uniform_in_support()

# define the Markov chains
mcs = [pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, mc_prop, start) for start in starts]

# run and delete burn-in
for mc in mcs:
    mc.run(10**3)
    mc.history.clear()

# run chains and use self adaptation
for mc in mcs:
    for i in range(100):
        mc.run(1000)
        mc.adapt()

# get the Markov chain data
mcmc_data = [mc.history[:] for mc in mcs]
stacked_data = np.vstack(mcmc_data)

# plot stacked_data
plt.figure(); plt.hexbin(stacked_data[:,0], stacked_data[:,1], cmap='gray_r')
plt.title('scalar')
plt.xlabel('Re{cS}')
plt.ylabel('Im{cS}')
plt.draw()


# find proposal function

# form "long_patches"
long_patches = pypmc.mix_adapt.r_value.make_r_gaussmix(mcmc_data)

# run variational bayes
vb = pypmc.mix_adapt.variational.GaussianInference(stacked_data[::50], initial_guess=long_patches, W0=np.diag([1e20]*dim))
vb_prune = .5*len(vb.data)/vb.K
vb.run(1000, abs_tol=1e-5, rel_tol=1e-10, prune=vb_prune, verbose=True)
vbmix = vb.make_mixture()

# plot vbmix
plt.figure(); plot_mixture(vbmix); plt.draw()


# run importance sampling
sampler = pypmc.sampler.importance_sampling.DeterministicIS(log_target, vbmix)
sampler.run(10**5)

# plot importance samples
plt.figure()
plt.hist2d(sampler.history[:][:,1], sampler.history[:][:,2], weights=sampler.history[:][:,0], cmap='gray_r', bins=100)

# calculate perplexity and ESS
print "Perplexity: %g" %perp(sampler.history[:][:,0])
print "ESS:        %g" %ess (sampler.history[:][:,0])

# calculate evidence
print "Z = %g +- %g" %(np.average(sampler.history[:][:,0]), np.sqrt(np.cov(sampler.history[:][:,0]) / len(sampler.history[:])))
