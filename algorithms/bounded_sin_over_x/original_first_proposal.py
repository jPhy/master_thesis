# for a new algorithm only change the part which is marked as such

from __future__ import division
import numpy as np

# need load_save and mcmc from '..'
import sys; sys.path.append('..')
from mcmc import create_mcmc_samples
from load_save import save_first_proposal

# load parameters
from params_original import params
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
N_perp_ess = 10**5
params.update( [('N_perp_ess', N_perp_ess), ('sys.argv', sys.argv)] )

# ############# never change a file which has already been executed #############



# ----------------------- replace algorithm below -----------------------

# patch chain output into short patches
mcmcmix = pypmc.tools.patch_data(np.vstack(values), L)


# group chains and create long patches
hierarchical_init = pypmc.mix_adapt.r_value.make_r_gaussmix(values, K_g, critical_r)


# apply hierarchical clustering
hc = pypmc.mix_adapt.hierarchical.Hierarchical(mcmcmix, hierarchical_init, verbose=True)
hc_converged = hc.run(kill=kill_in_hc)
if hc_converged:
    print 'hierarchical clustering converged in step %i\n' %(hc_converged)
else:
    print 'hierarchical clustering did not converge\n'
reduced_proposal = hc.g

if abandon_weights:
    reduced_proposal.weights[:] = 1.
    reduced_proposal.normalize()


print 'have %i live components after hierarchical clustering\n' %(reduced_proposal.weights!=0).sum()

# save number of chain groups; if and in which step hierarchical clustering converged; number of components after hierarchical
params.update( [('hc_converge_step', hc_converged), ('final_number_of_components', len(reduced_proposal))] )

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
