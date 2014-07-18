import pypmc, eos, numpy as np
from pypmc.tools import plot_mixture
from pypmc.tools.convergence import perp, ess

# define log posterior function
constraints = ["B^0_s->mu^+mu^-::BR@LHCb-2013D", "B^0_s->mu^+mu^-::BR@CMS-2013B"]
priors = [eos.LogPrior.Flat("Re{cP}", range_min=-15, range_max=15),
          eos.LogPrior.Flat("Im{cP}", range_min=-15, range_max=15)]
dim = len(priors)
ind = pypmc.tools.indicator.hyperrectangle([-15]*2, [15]*2)
options = {"scan-mode": "cartesian", "model": "WilsonScan", "form-factors": "BZ2004"}

ana = eos.Analysis(constraints, priors, options)

# merge with indicator
log_target = pypmc.tools.indicator.merge_function_with_indicator(ana, ind, -np.inf)


# Markov chain prerun --> run 10 Markov chains with different initial points

# define a proposal for the initial Markov chain run
mc_prop = pypmc.density.gauss.LocalGauss(np.eye(dim)*.1)

# define initial points for the Markov chain run
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
    for i in range(100):
        mc.run(1000)
        mc.adapt()

# get the Markov chain data
mcmc_data = [mc.history[:] for mc in mcs]
stacked_data = np.vstack(mcmc_data)

# plot stacked_data
plt.figure(); plt.hexbin(stacked_data[:,0], stacked_data[:,1], cmap='gray_r')
plt.title('pseudoscalar')
plt.xlabel('Re{cP}')
plt.ylabel('Re{cP}')
plt.draw()


# find proposal function

# form "short_patches"
short_patches = pypmc.tools.patch_data(stacked_data, L=50)
inimix = pypmc.density.mixture.MixtureDensity(short_patches.components[::1000])

# run variational bayes
vb = pypmc.mix_adapt.variational.GaussianInference(stacked_data[::50], initial_guess=inimix)
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
