#! /usr/bin/python
import pypmc
from pypmc.tools import plot_mixture
import numpy as np
from matplotlib import pyplot as plt

# choose a dimension
dim = 20

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
        mc.run(5000) # TODO less samples to see non prune in PMC
        mc.adapt()

# get the Markov chain data
mcmc_data = [mc.history[:] for mc in mcs]
stacked_data = np.vstack(mcmc_data)

# plot stacked_data (pure Markov Chain data)
plt.figure(); plt.hexbin(stacked_data[:,0], stacked_data[:,1], cmap='gray_r', extent=(-30, 30, -20, 6))
# optional: plot autocorrelation of first chain
# plt.figure(); plt.acorr(mcmc_data[0][:,0] - mcmc_data[0][:,0].mean(), maxlags=1000)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('plain_mc_data_5000.png') # not .svg because there too many hexagin objects

print 'Markov Chains done'


# form the "long_patches"
long_patches = pypmc.mix_adapt.r_value.make_r_gaussmix(mcmc_data)


# form first proposal with PMC
print 'running PMC'
pmcmix = pypmc.mix_adapt.pmc.gaussian_pmc(stacked_data[::100], long_patches, copy=True)
pmcmix.prune(.5/len(long_patches))
for i in range(1000-1):
    print i
    pypmc.mix_adapt.pmc.gaussian_pmc(stacked_data[::100], pmcmix, copy=False)
    pmcmix.prune(.5/len(long_patches))
plt.figure()
plot_mixture(pmcmix)
plt.xlim(-30,30)
plt.ylim(-20,6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('first_prop_pmc_5000.pdf')
print 'PMC done'


# form first proposal with Variational Bayes

# run variational bayes with samples --> use the long patches as initial guess
vb = pypmc.mix_adapt.variational.GaussianInference(stacked_data[::100], initial_guess=long_patches)
print 'running VB...'
vb.run(1000, abs_tol=1e-5, rel_tol=1e-10, prune=.5*len(vb.data)/vb.K, verbose=False)
print 'VB done'
vbmix = vb.make_mixture()
plt.figure()
plot_mixture(vbmix)
plt.xlim(-30,30)
plt.ylim(-20,6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('first_prop_vb_5000.pdf')


# form first proposal with Hierarchical Clustering

# for hierarchical clustering (and VBMerge) form "short_patches"
short_patches = pypmc.tools.patch_data(stacked_data, L=100)

# run hierarchical clustering
hc = pypmc.mix_adapt.hierarchical.Hierarchical(short_patches, long_patches, verbose=True)
print 'running HC...'
hc.run()
print 'HC done'
hcmix = hc.g
plt.figure()
plot_mixture(hcmix)
plt.xlim(-30,30)
plt.ylim(-20,6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('first_prop_hc_5000.pdf')
