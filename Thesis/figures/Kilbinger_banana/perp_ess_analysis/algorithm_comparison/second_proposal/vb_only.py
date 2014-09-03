#! /usr/bin/python
import pypmc
from pypmc.tools.convergence import perp, ess
import numpy as np

# choose a dimension
dim = 20

# define number of samples to estimate perp and ess
N_perp_ess = 100000

# define number of samples for vb2
vb2_N = 6000

# load target function
from banana_function import LogTarget; log_target = LogTarget(dim)

print 'running Markov Chains'
# Markov chain prerun --> run 10 Markov chains with different initial points

# define a proposal for the initial Markov chain run:
# local Gauss with 0.1 * "unit matrix" as covariance
mc_prop = pypmc.density.gauss.LocalGauss(np.eye(dim)*.1)

# define initial points for the Markov chain run
# we will draw samples from the uniform distribution between [-5]*dim and [+5]*dim
starts = np.random.uniform(-5.,5.,size=10*dim).reshape((10,dim))
# log_target(starts[i]) must not be -inf!
for start in starts:
    while log_target(start) == -np.inf:
        start[:] = np.random.uniform(-5.,5.,size=dim)

# define the Markov chains
mcs = [pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, mc_prop, start) for start in starts]

# run and delete burn-in
for mc in mcs:
    mc.run(10**3)
    mc.history.clear()

# run chains and use self adaptation
for mc in mcs:
    for i in range(20):
        mc.run(5000)
        mc.adapt()

# get the Markov chain data
mcmc_data = [mc.history[:] for mc in mcs]
stacked_data = np.vstack(mcmc_data)

print 'Markov Chains done'


# form the "long_patches"
long_patches = pypmc.mix_adapt.r_value.make_r_gaussmix(mcmc_data)
chain_groups = len(long_patches) / 15 # components per goup = 15

# form first proposal with Variational Bayes

# run variational bayes with samples --> use the long patches as initial guess
vb = pypmc.mix_adapt.variational.GaussianInference(stacked_data[::100], initial_guess=long_patches)
print 'running VB1...'
vb.run(1000, abs_tol=1e-5, rel_tol=1e-10, prune=.5*len(vb.data)/vb.K, verbose=True)
vbmix = vb.make_mixture()
# calculate perp/ess
vb_sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, vbmix)
vb_sampler.run(N_perp_ess, trace_sort=False)
vb_weighted_samples = vb_sampler.history[-1]
vb_weights = vb_weighted_samples[:,0]
vb_perp = perp(vb_weights)
vb_ess  = ess (vb_weights)
components_vb = len(vbmix)
print 'VB1 done'


print 'running VB2...'
prior_for_vb2 = vb.posterior2prior()
prior_for_vb2.pop('alpha0')
vb2 = pypmc.mix_adapt.variational.GaussianInference(vb_weighted_samples[:vb2_N,1:], weights=vb_weights[:vb2_N],
                                                    initial_guess=vbmix, **prior_for_vb2)
vb2.run(1000, abs_tol=1e-5, rel_tol=1e-10, verbose=True)
vb2mix = vb2.make_mixture()
vb2_sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, vb2mix)
vb2_sampler.run(N_perp_ess, trace_sort=False)
vb2_weighted_samples = vb2_sampler.history[-1]
vb2_weights = vb2_weighted_samples[:,0]
vb2_perp = perp(vb2_weights)
vb2_ess  = ess (vb2_weights)
components_vb2 = len(vb2mix)
N_importance_to_adapt = vb2.N
print 'VB2 done'


# append this run's results to output file
outfile = open('vb_only.txt', 'a')
outfile.write('\n' + str(chain_groups) + ' ' + str(components_vb) + ' ' + str(vb_perp) + ' ' + str(vb_ess) + ' ' + str(components_vb2) + ' ' + str(vb2_perp) + ' ' + str(vb2_ess) + ' ' + str(N_importance_to_adapt))
outfile.close()
