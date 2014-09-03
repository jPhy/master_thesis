#! /usr/bin/python
import pypmc
from pypmc.tools.convergence import perp, ess
import numpy as np

# choose a dimension
dim = 20

# define number of samples per component
N_c = 600

# define number of samples to estimate perp and ess
N_perp_ess = 100000

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

# for hierarchical clustering (and VBMerge) form "short_patches"
short_patches = pypmc.tools.patch_data(stacked_data, L=100)

# run hierarchical clustering
hc = pypmc.mix_adapt.hierarchical.Hierarchical(short_patches, long_patches, verbose=True)
print 'running HC...'
hc.run()
hcmix = hc.g
hc_sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, hcmix)
hc_sampler.run(N_perp_ess, trace_sort=True)
hc_weighted_samples = hc_sampler.history[-1]
hc_weights = hc_weighted_samples[:,0]
hc_perp = perp(hc_weights)
hc_ess  = ess (hc_weights)
components_hc = len(hcmix)
print 'HC done'


# importance sampling main loop

sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, hcmix)
previous_perp = - np.inf
converge_step = None

for i in range(25):
    print "step", i

    # run len(sampler.proposal) * N_c steps
    latent = sampler.run(N_c * len(sampler.proposal), trace_sort=True)

    # get the weighted samples that have just been generated
    weighted_samples = sampler.history[-1]

    # calculate perplexity
    this_perp = pypmc.tools.convergence.perp(weighted_samples[:,0])

    # stop if convergence criterion reached
    if (this_perp - previous_perp) / this_perp < .05:
        converge_step = i
        print 'PMC converged in step ' + str(i)
        break
    else:
        previous_perp = this_perp

    # update the proposal using PMC
    pypmc.mix_adapt.pmc.gaussian_pmc(weighted_samples[:,1:], sampler.proposal, weights=weighted_samples[:,0],
                                     latent=latent, rb=True, mincount=20, copy=False)

    # can prune here because samples are not reused
    sampler.proposal.prune()

    print 'have %i live components PMC step %i\n' %(len(sampler.proposal), i)

# final run
final_number_of_components = len(sampler.proposal)
N_importance_to_adapt = len(sampler.history[:])
sampler.run(N_perp_ess, trace_sort=True)
weighted_samples = sampler.history[-1]
last_perp = pypmc.tools.convergence.perp(weighted_samples[:,0])
last_ess  = pypmc.tools.convergence.ess (weighted_samples[:,0])

# append this run's results to output file
outfile = open('original_algorithm_15_components.txt', 'a')
outfile.write('\n' + str(chain_groups) + ' ' + str(components_hc) + ' ' + str(hc_perp) + ' ' + str(hc_ess) + ' ' + str(final_number_of_components) + ' ' + str(last_perp) + ' ' + str(last_ess) + ' ' + str(N_importance_to_adapt))
outfile.close()
