from __future__ import division
from function import gauss_shell
from pypmc.tools.convergence import perp, ess
from pypmc.density.mixture import create_gaussian_mixture
from pypmc.tools import plot_mixture
import matplotlib.pyplot as plt
import numpy as np
import os
import pypmc

# create folder for this run
i = 0
while os.path.exists('./run' + str(i)):
    i += 1
dirname = 'run' + str(i)
dirpath = './' + dirname + '/'
os.makedirs(dirname)


# set parameters
params = dict(
random_seed = np.random.randint(2**62),
dim = 2,
kill_in_hc = True,
k = 10, # number of markov chains
N_burn_in = 10**3, # number of burn-in samples
N_MCMC = 10**4, # number of steps per chain
N_adapt = 500, # adapt the Markov chains every N_adapt steps (not during burn-in)
K_g = 15, # components per group
L = 100,
critical_r = 1.2,
N_c = 200, # number of sampels per component
)

# set these variables
locals().update(params)

# seed rng
np.random.seed(random_seed)

# save configuration in a file
with open(dirpath + 'config', 'w') as configfile:
    configfile.write('original_algorithm\n')
    configfile.write(str(params))

# write status messages to a file
statusfile = open(dirpath + 'runtime_messages', 'w')

# save plots to this file
from matplotlib.backends.backend_pdf import PdfPages
plotfile = PdfPages(dirpath + 'plots.pdf')

# define the target density
# here, two Gaussian shells with parameters
c_0 = np.array([+3.5] + (dim-1)*[0]); c_1 = np.array([-3.5] + (dim-1)*[0]); r = 2.0; w = 0.1
# and equal weight are used
def log_target(x):
    return np.log(  .5 * (gauss_shell(x, c_0, w, r) + gauss_shell(x, c_1, w, r))  )

# define the initial proposal for the local random walk markov chain
prop_sigma = np.eye(dim) * .1
prop = pypmc.density.gauss.LocalGauss(prop_sigma)

# choose random initializations to run k chains
starts = []
for i in range(k):
    point = np.random.rand(dim)*6. - 3.
    while np.isinf(log_target(point)):
        point = np.random.rand(dim)*6. - 3.
    starts.append(point)

# define the markov chain objects
mcs = [pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, prop, starts[i]) for i in range(k)]

# run burn-in
for i, mc in enumerate(mcs):
    print "burn in chain", i
    mc.run(N_burn_in)

# delete burn-in from history
for mc in mcs:
    mc.history.clear()

# run N_MCMC steps adapting the proposal every N_adapt steps
# hereby save the accept count which is returned by mc.run
accept_counts = np.zeros(k)
for n, mc in enumerate(mcs):
    print "running chain", n
    for i in range(N_MCMC//N_adapt):
        accept_counts[n] += mc.run(N_adapt)
        mc.adapt()

# extract a reference to the history of all visited points
values = [mcs[i].history[:] for i in range(k)]
accept_rates = [float(accept_counts[i]) / len(values[i]) for i in range(k)]
for i, count in enumerate(accept_counts):
    print "Chain %i accepted %4.2f%% of the proposed points" % (i, accept_rates[i] * 100)
    statusfile.write("Chain %i accepted %4.2f%% of the proposed points\n" % (i, accept_rates[i] * 100))

# patch chain output
patches = []
for val in values:
    patches.extend([val[patch_start:patch_start + L] for patch_start in range(0, len(val), L)])
patches = np.array(patches)

means   = np.array([patch.mean(axis=0) for patch in patches])
covs    = np.array([np.cov(patch, rowvar=0) for patch in patches])
mcmcmix_components = []
for mean, cov in zip(means, covs):
    try:
        mcmcmix_components.append(pypmc.density.gauss.Gauss(mean, cov))
    except np.linalg.LinAlgError:
        cov = np.diag(np.diag(cov))
        try:
            mcmcmix_components.append(pypmc.density.gauss.Gauss(mean, cov))
        except np.linalg.LinAlgError:
            print "Could not create component:"
            print 'mean:', mean
            print 'cov:', cov
            statusfile.write("Could not create component:\n")
            statusfile.write("mean: %s\n" %mean)
            statusfile.write("cov: %s\n" %cov)
    # delete components with negative det
    if mcmcmix_components[-1].det_sigma <= 0.:
        mcmcmix_components.pop()
mcmcmix = pypmc.density.mixture.MixtureDensity(mcmcmix_components)

plt.figure()
plt.title('from chains')
plot_mixture(mcmcmix, 0,1)
plotfile.savefig()

# --------------------------- end of markov chain ---------------------------------------

# --------------------------- chain grouping --------------------------------------------

chain_groups = pypmc.mix_adapt.r_value.r_group([np.mean(mc.history[:], axis=0) for mc in mcs],
                                               [np.cov(mc.history[:], rowvar=0) for mc in mcs],
                                               len(val), critical_r)

statusfile.write('found %i chain grous\n' %len(chain_groups) )

long_patches_means = []
long_patches_covs = []
for group in chain_groups:
    # we want K_g components from k_g = len(group) chains
    k_g = len(group)
    if K_g >= k_g:
        # find minimal lexicographic integer partition
        n = [K_g // k_g for i in range(k_g)]
        remainder = K_g % k_g
        for i in range(remainder):
            n[i] += 1
        for i, chain_index in enumerate(group):
            # need to partition in n[i] parts
            data_full_chain = mcs[chain_index].history[:]
            chain_length = len(data_full_chain)
            # find minimal lexicographic integer partition of chain_length into n[i]
            this_patch_lengths = [chain_length // n[i] for j in range(n[i])]
            remainder = chain_length % n[i]
            for j in range(remainder):
                this_patch_lengths[j] += 1
            start = 0
            stop  = this_patch_lengths[0]
            for next_len in this_patch_lengths:
                this_data = data_full_chain[start:stop]
                long_patches_means.append( np.mean(this_data, axis=0) )
                long_patches_covs.append ( np.cov (this_data, rowvar=0) )
                start += next_len
                stop  += next_len
    else:
        # form one long chain and set k_g = 1
        k_g = 1
        # make one large chain
        data_full_chain = np.vstack([mcs[i].history[:] for i in group])
        chain_length = len(data_full_chain)
        # need to partition into K_g parts -- > minimal lexicographic integer partition
        this_patch_lengths = [chain_length // K_g for j in range(K_g)]
        remainder = chain_length % K_g
        for j in range(remainder):
            this_patch_lengths[j] += 1
        start = 0
        stop  = this_patch_lengths[0]
        for next_len in this_patch_lengths:
                this_data = data_full_chain[start:stop]
                long_patches_means.append( np.mean(this_data, axis=0) )
                long_patches_covs.append ( np.cov (this_data, rowvar=0) )
                start += next_len
                stop  += next_len


hierarchical_init = create_gaussian_mixture(long_patches_means, long_patches_covs)

plt.figure()
plt.title('hierarchical init')
plot_mixture(hierarchical_init, 0,1)
plotfile.savefig()

# ----------------------- hierarchical clustering --------------------------------------

hc = pypmc.mix_adapt.hierarchical.Hierarchical(mcmcmix, hierarchical_init, verbose=True)
hc_converged = hc.run(kill=kill_in_hc)
if hc_converged:
    statusfile.write('hierarchical clustering converged in step %i\n' %(hc_converged) )
else:
    statusfile.write('hierarchical clustering did not converge\n')
reduced_proposal = hc.g

# cannot trust component weights from Markov chain --> make them uniform
reduced_proposal.weights[:] = 1.
reduced_proposal.normalize()

plt.figure()
plt.title('first glance')
plot_mixture(reduced_proposal, 0,1)
plotfile.savefig()

statusfile.write('have %i live components after hierarchical clustering\n' %(reduced_proposal.weights!=0).sum())

# ---------------------- continue with importance sampling ----------------------

# define an ImportanceSampler object using ``reduced_proposal``
sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, reduced_proposal)

generating_components = []
perplexities = [0.]

for i in range(10):
    print "step", i
    # run 10,000 steps
    generating_components.append(sampler.run(N_c * (sampler.proposal.weights!=0).sum(), trace_sort=True))
    # get the weighted samples that have just been generated
    weighted_samples = sampler.history[-1]
    perplexities.append(perp(weighted_samples[:,0]))
    if (perplexities[i+1] - perplexities[i]) / perplexities[i+1] < .05:
        statusfile.write('PMC converged in step ' + str(i) + '\n')
        break
    # update the proposal using PMC
    pypmc.mix_adapt.pmc.gaussian_pmc(weighted_samples[:,1:], sampler.proposal, weights=weighted_samples[:,0],
                                     latent=generating_components[-1], rb=True, mincount=20, copy=False)
    # plot
    plt.figure()
    plt.title('proposal after PMC update ' + str(i))
    plot_mixture(sampler.proposal, 0,1, cutoff=.01)
    plotfile.savefig()
    statusfile.write('have %i live components PMC step %i\n' %((sampler.proposal.weights!=0).sum(), i)    )

statusfile.write('Perplexities:\n')
for perp in perplexities[1:]: #[1:] because of initial 0.
    statusfile.write('%s\n' %perp)

statusfile.write('Effective sample sizes:\n')
for ws in sampler.history:
    statusfile.write('%s\n' %ess(ws[:,0]))

statusfile.close()
plotfile.close()
