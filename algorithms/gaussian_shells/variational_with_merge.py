# for a new algorithm only change the part which is marked as such

from __future__ import division
import numpy as np

# need load_save and mcmc from '..'
import sys; sys.path.append('..')
from mcmc import create_mcmc_samples
from load_save import save_first_proposal

# load parameters
from params_variational_with_merge import params
locals().update(params)

# load the target function
from target import LogTarget; log_target = LogTarget(dim)

# load pypmc
import pypmc
from pypmc.tools.convergence import perp, ess
from pypmc.density.mixture import create_gaussian_mixture

# create data from markov-chains
values = create_mcmc_samples(log_target, k, N_burn_in, N_MCMC, N_adapt)

# ***********************************************************************
# ****************** nothing above should be changed ! ******************
# ***********************************************************************


# developer notes:
# save all parameter you want to save as:
# params.update( (('param_1', value_1), (param_2, value_2)) )
# params will be written into the database and into the proposal as
# attribute `creation_parameters`
# example: save the number of samples used for the perp and ess estimate; save the command line call
N_perp_ess = 10**4
params.update( [('N_perp_ess', N_perp_ess), ('sys.argv', sys.argv)] )

# ############# never change a file which has already been executed #############



# ----------------------- replace algorithm below -----------------------

# thin data known to variational bayes
data = np.vstack(values)[::N_thin]

# patch chain output into short patches
short_patches = []
for val in values:
    short_patches.extend([val[patch_start:patch_start + L] for patch_start in range(0, len(val), L)])
short_patches = np.array(short_patches)

means   = np.array([patch.mean(axis=0) for patch in short_patches])
covs    = np.array([np.cov(patch, rowvar=0) for patch in short_patches])
short_mcmcmix_components = []
for mean, cov in zip(means, covs):
    try:
        short_mcmcmix_components.append(pypmc.density.gauss.Gauss(mean, cov))
    except np.linalg.LinAlgError:
        cov = np.diag(np.diag(cov))
        try:
            short_mcmcmix_components.append(pypmc.density.gauss.Gauss(mean, cov))
        except np.linalg.LinAlgError:
            print "Could not create component:"
            print 'mean:', mean
            print 'cov:', cov
    # delete components with negative det
    if short_mcmcmix_components[-1].det_sigma <= 0.:
        short_mcmcmix_components.pop()
short_mcmcmix = pypmc.density.mixture.MixtureDensity(short_mcmcmix_components)

# group chains and create long patches
chain_groups = pypmc.mix_adapt.r_value.r_group([np.mean(chain_values[:], axis=0) for chain_values in values],
                                               [np.cov(chain_values[:], rowvar=0) for chain_values in values],
                                               len(values[0]), critical_r)

print 'found %i chain grous\n' %len(chain_groups)

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
            data_full_chain = values[chain_index]
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
        data_full_chain = np.vstack([values[i] for i in group])
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
long_mcmcmix = create_gaussian_mixture(long_patches_means, long_patches_covs)

# set vb_prune
vb_prune = .5 * len(data)/len(long_mcmcmix)
params.update((('vb_prune', vb_prune),))

# run VBMerge
vbm = pypmc.mix_adapt.variational.VBMerge(short_mcmcmix, len(data), initial_guess=long_mcmcmix)
if vbm_prune:
    vbm_converged = vbm.run(N_max_vbm, verbose=True, rel_tol=vb_rel_tol, abs_tol=vb_abs_tol, prune=vb_prune)
else:
    vbm_converged = vbm.run(N_max_vbm, verbose=True, rel_tol=vb_rel_tol, abs_tol=vb_abs_tol, prune=0.)
print('vbm completed\n\n\n')

# run vb
vb = pypmc.mix_adapt.variational.GaussianInference(data, **vbm.prior_posterior())
vb_converged = vb.run(N_max_vb, verbose=True, rel_tol=vb_rel_tol, abs_tol=vb_abs_tol, prune=vb_prune)
reduced_proposal = vb.make_mixture()

if abandon_weights:
    reduced_proposal.weights[:] = 1.
    reduced_proposal.normalize()

# save:
#    final number of components
#    number of components after vbm
#    number of components after vb
#    if and in which step variational bayes converged
#    if and in which step VBMerge converged
params.update( [('final_number_of_components', len(reduced_proposal)), ('number_of_components_after_vb', vb.K),
                ('number_of_components_after_vbm', vbm.K), ('vb_converge_step', vb_converged),
                ('vbm_converge_step', vbm_converged)] )

# ----------------------- replace algorithm above -----------------------

# ***********************************************************************
# ****************** nothing below should be changed ! ******************
# ***********************************************************************





# use importance sampling to calculate perplexity and effective sample size

# define an ImportanceSampler object using ``reduced_proposal``
sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, reduced_proposal)

# run N_perp_ess steps
sampler.run(N_perp_ess, trace_sort=True)

# get the weights from the samples that have just been generated
weighted_samples = sampler.history[-1]
weights = weighted_samples[:,0]

# calculate perplexity and ess
perplexity = perp(weights)
ess        = ess (weights)

# save perplexity and ess
params.update( [('perplexity', perplexity), ('ess', ess)] )

# dump results
save_first_proposal(reduced_proposal, params)
