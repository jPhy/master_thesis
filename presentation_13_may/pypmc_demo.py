import pypmc
from pypmc.tools import plot_mixture

# choose a dimension
dim = 2

# choose a target function
from gaussian_shells import LogTarget; log_target = LogTarget(dim)

# Markov chain prerun --> run 10 Markov chains with different initial points

# define a proposal for the initial Markov chain run
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
        mc.run(1000)
        mc.adapt()

# get the Markov chain data
mcmc_data = [mc.history[:] for mc in mcs]
stacked_data = np.vstack(mcmc_data)

# plot stacked_data
plt.figure(); plt.hexbin(stacked_data[:,0], stacked_data[:,1], cmap='gray_r')
# optional: plot autocorrelation of first chain
# plt.figure(); plt.acorr(mcmc_data[0][:,0] - mcmc_data[0][:,0].mean(), maxlags=1000)
plt.draw()

# form the "long_patches"
long_patches = pypmc.mix_adapt.r_value.make_r_gaussmix(mcmc_data)

# run variational bayes with samples --> use the long patches as initial guess
vb = pypmc.mix_adapt.variational.GaussianInference(stacked_data[::50], initial_guess=long_patches)
vb.run(1000, abs_tol=1e-5, rel_tol=1e-10, prune=.5*len(vb.data)/vb.K, verbose=True)
vbmix = vb.make_mixture()

# for hierarchical clustering and VBMerge form "short_patches"
short_patches = pypmc.tools.patch_data(stacked_data, L=100)

# run variational bayes with mixture --> use the long patches as initial guess
merge = pypmc.mix_adapt.variational.VBMerge(short_patches, len(stacked_data), initial_guess=long_patches)
merge.run(1000, abs_tol=1e-5, rel_tol=1e-10, prune=.5*len(vb.data)/vb.K, verbose=True)
mergemix = merge.make_mixture()

# run hierarchical clustering
hc = pypmc.mix_adapt.hierarchical.Hierarchical(short_patches, long_patches, verbose=True)
hc.run()
hcmix = hc.g

# plot the mixtures from these algorithms
plt.figure(); plt.title('Hierarchical'); plot_mixture(hc.g)
plt.figure(); plt.title('VB samples'); plot_mixture(vbmix)
plt.figure(); plt.title('VB mixtures'); plot_mixture(mergemix)
plt.draw()

# define an importance sampler for each proposal and run 10**5 steps
hc_sampler = pypmc.sampler.importance_sampling.DeterministicIS(log_target, hcmix, std_weights=True)
vb_sampler = pypmc.sampler.importance_sampling.DeterministicIS(log_target, vbmix, std_weights=True)
merge_sampler = pypmc.sampler.importance_sampling.DeterministicIS(log_target, mergemix, std_weights=True)
hc_sampler.run(10**5)
vb_sampler.run(10**5)
merge_sampler.run(10**5)

# plot importance samples
plt.figure(); plt.title('Hierarchical')
plt.hist2d(hc_sampler.history[:][:,1], hc_sampler.history[:][:,2], weights=hc_sampler.history[:][:,0], cmap='gray_r', bins=100)

plt.figure(); plt.title('VB samples')
plt.hist2d(vb_sampler.history[:][:,1], vb_sampler.history[:][:,2], weights=vb_sampler.history[:][:,0], cmap='gray_r', bins=100)

plt.figure(); plt.title('VB mixtures')
plt.hist2d(merge_sampler.history[:][:,1], merge_sampler.history[:][:,2], weights=merge_sampler.history[:][:,0], cmap='gray_r', bins=100)

plt.draw()

print "hc perplexity", pypmc.tools.convergence.perp(hc_sampler.history[:][:,0])
print "hc ESS", pypmc.tools.convergence.ess(hc_sampler.history[:][:,0])
print "vb perplexity", pypmc.tools.convergence.perp(vb_sampler.history[:][:,0])
print "vb ESS", pypmc.tools.convergence.ess(vb_sampler.history[:][:,0])
print "merge perplexity", pypmc.tools.convergence.perp(merge_sampler.history[:][:,0])
print "merge ESS", pypmc.tools.convergence.ess(merge_sampler.history[:][:,0])








# run a second variational bayes to get the component weights right

# get the full posterior from vb
mcmc_vb_posterior = vb.posterior2prior()

# the component weights cannot be trusted --> remove them, i.e. take default non-informative
prior_for_vb2 = mcmc_vb_posterior.copy()
prior_for_vb2.pop('alpha0')

# create the new variational instance and run it
# Note: ``prune`` is now set to default (1 effective sample per component) because all pruning should already have been done
vb2 = pypmc.mix_adapt.variational.GaussianInference(vb_sampler.history[:][:,1:], weights=vb_sampler.history[:][:,0],
                                                    initial_guess=vb_sampler.proposal, **prior_for_vb2)
vb2.run(1000, abs_tol=1e-5, rel_tol=1e-10, verbose=True)

# replace old proposal by new one with corrected component weights
vb_sampler.proposal = vb2.make_mixture()

vb_sampler.run(10**5)

print "old vb perplexity", pypmc.tools.convergence.perp(vb_sampler.std_weights[0])
print "old vb ESS", pypmc.tools.convergence.ess(vb_sampler.std_weights[0])
print "new vb perplexity", pypmc.tools.convergence.perp(vb_sampler.std_weights[1])
print "new vb ESS", pypmc.tools.convergence.ess(vb_sampler.std_weights[1])
print "combined vb perplexity", pypmc.tools.convergence.perp(vb_sampler.history[:][:,0])
print "combined vb ESS", pypmc.tools.convergence.ess(vb_sampler.history[:][:,0])


print "Total effective samples with Cornuet:", pypmc.tools.convergence.ess(vb_sampler.history[:][:,0]) * len(vb_sampler.history[:])
print "Total effective samples without Cornuet:", pypmc.tools.convergence.ess(vb_sampler.std_weights[0]) * len(vb_sampler.history[0]) + pypmc.tools.convergence.ess(vb_sampler.std_weights[1]) * len(vb_sampler.history[1])



# optional: to call pmc
# pypmc.mix_adapt.pmc.gaussian_pmc(hc_sampler.history[:][:,1:], hcmix, hc_sampler.history[:][:,0], copy=False)
