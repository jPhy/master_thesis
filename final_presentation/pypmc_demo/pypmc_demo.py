import pypmc
from pypmc.tools import plot_mixture
from vb_update_movie import vb_update_movie

dim = 4

# target function
from two_bananas import LogTarget; log_target = LogTarget(dim)

# Markov chain prerun --> run 2 Markov chains, one in each mode

# define initial proposal for the Markov chain run
mc_prop = pypmc.density.gauss.LocalGauss(np.eye(dim)*.1)

# define initial points for the Markov chain run (one in each mode)
starts = [
              np.array([  2.21400968,   6.37248868, -11.88389791,  -5.91889878]),
              np.array([-10.62816039,  -3.2020552 ,   0.04447288,   5.3784658 ])
         ]

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
plt.figure()
plt.subplot(211)
plt.title('Markov chain 1')
plt.hexbin(mcmc_data[0][:,1], mcmc_data[0][:,2], cmap='gray_r', extent=(-30,10,-30,30))
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
plt.subplot(212)
plt.title('Markov chain 2')
plt.hexbin(mcmc_data[1][:,1], mcmc_data[1][:,2], cmap='gray_r', extent=(-30,10,-30,30))
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
plt.figure()
plt.title('Markov chain - combination')
plt.hexbin(stacked_data[:,1], stacked_data[:,2], cmap='gray_r', extent=(-30,10,-30,30))
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
plt.draw()



# form the "long_patches"
long_patches = pypmc.mix_adapt.r_value.make_r_gaussmix(mcmc_data)

# run variational bayes with samples --> use the long patches as initial guess
vb = pypmc.mix_adapt.variational.GaussianInference(stacked_data[::50], initial_guess=long_patches)
plt.figure()
vb_update_movie(vb, 1000, abs_tol=1e-5, rel_tol=1e-10, prune=.5*len(vb.data)/vb.K, verbose=True)
vbmix = vb.make_mixture()

# define an importance sampler for each proposal and run 10**5 steps
vb_sampler = pypmc.sampler.importance_sampling.DeterministicIS(log_target, vbmix, std_weights=True)
vb_sampler.run(10**5)

# plot importance samples
plt.figure()
plt.title('Importance Samples')
plt.hist2d(vb_sampler.history[:][:,2], vb_sampler.history[:][:,3], weights=vb_sampler.std_weights[:][:,0], cmap='gray_r', bins=100)
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
plt.xlim(-30,10)
plt.ylim(-30,30)
plt.draw()

%cpaste
print "vb perplexity", pypmc.tools.convergence.perp(vb_sampler.history[:][:,0])
print "vb ESS", pypmc.tools.convergence.ess(vb_sampler.history[:][:,0])
--








# run a second variational bayes to get the component weights right

# get the full posterior from vb
mcmc_vb_posterior = vb.posterior2prior()

# the component weights cannot be trusted --> remove them, i.e. take default non-informative
prior_for_vb2 = mcmc_vb_posterior.copy()
prior_for_vb2.pop('alpha0')

# create the new variational instance and run it
# use at most 10**4 samples from importance sampling
# Note: ``prune`` is now set to default (1 effective sample per component) because all pruning should already have been done
vb2 = pypmc.mix_adapt.variational.GaussianInference(vb_sampler.history[:][:,1:][:10**4],
                                                    weights=vb_sampler.std_weights[:][:,0][:10**4],
                                                    initial_guess=vb_sampler.proposal, **prior_for_vb2)
vb2.run(1000, abs_tol=1e-5, rel_tol=1e-10, verbose=True)

# replace old proposal by new one with corrected component weights
vb_sampler.proposal = vb2.make_mixture()

vb_sampler.run(10**5)


plt.figure()
plt.title('improved Importance Samples')
plt.hist2d(vb_sampler.history[1][:,2], vb_sampler.history[1][:,3], weights=vb_sampler.std_weights[1][:,0], cmap='gray_r', bins=100)
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
plt.xlim(-30,10)
plt.ylim(-30,30)
plt.draw()


%cpaste
print "old vb perplexity", pypmc.tools.convergence.perp(vb_sampler.std_weights[0])
print "old vb ESS", pypmc.tools.convergence.ess(vb_sampler.std_weights[0])
print "new vb perplexity", pypmc.tools.convergence.perp(vb_sampler.std_weights[1])
print "new vb ESS", pypmc.tools.convergence.ess(vb_sampler.std_weights[1])
print "combined vb perplexity", pypmc.tools.convergence.perp(vb_sampler.history[:][:,0])
print "combined vb ESS", pypmc.tools.convergence.ess(vb_sampler.history[:][:,0])


print "Total effective samples with Cornuet:", pypmc.tools.convergence.ess(vb_sampler.history[:][:,0]) * len(vb_sampler.history[:])
print "Total effective samples without Cornuet:", pypmc.tools.convergence.ess(vb_sampler.std_weights[0]) * len(vb_sampler.history[0]) + pypmc.tools.convergence.ess(vb_sampler.std_weights[1]) * len(vb_sampler.history[1])
--
