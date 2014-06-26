#! /usr/bin/python
import numpy as np
import pypmc

# define a proposal
prop_sigma = np.eye(1)*.01
prop = pypmc.density.gauss.LocalGauss(prop_sigma)

# define the target; i.e., the function you want to sample from.
# In this case, it is a Gaussian with mean "target_mean" and
# covariance "target_sigma".
#
# Note that the target function "log_target" returns the log of the
# unnormalized gaussian density.
target_sigma = np.array([[0.1]])
target_mean0 = np.array([-5.])
target_mean1 = np.array([+5.])

target_mix = pypmc.density.mixture.create_gaussian_mixture([target_mean0, target_mean1], [target_sigma]*2)

log_target = target_mix.evaluate

# choose an initialization
start = np.array([+5.])

# define the markov chain object
mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, prop, start, prealloc=10**6)

# run burn-in
mc.run(10**4)

# delete burn-in from history
mc.history.clear()

accept_count = 0
# run 100,000 steps adapting the proposal every 500 steps
for i in range(200):
    accept_count += mc.run(500)
    mc.adapt()

# extract a reference to the history of all visited points
values = mc.history[:]
accept_rate = float(accept_count) / len(values)
print("The chain accepted %4.2f%% of the proposed points" % (accept_rate * 100) )

# plot the result
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('For plotting "matplotlib" needs to be installed')
    exit(1)

points = np.linspace(-7,7, 10**5).reshape((10**5,1))
target_values = 2. * np.exp(target_mix.multi_evaluate(points))
plt.plot(points[:,0], target_values)

plt.hist(values[:,0], color='black', bins=20, normed=True)

plt.xlabel('$x$')
plt.ylabel('$P(x)$')

plt.savefig('figure.svg')
