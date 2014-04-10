# for a new algorithm only change the part which is marked as such

from __future__ import division
import numpy as np

# need load_save and mcmc from '..'
import sys; sys.path.append('..')
from mcmc import create_mcmc_samples
from load_save import save_first_proposal

# load parameters
from params import params
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

# thin data
data = np.vstack(values)[::N_thin]


# use variational bayes choosing an initialization scheme from the parameters

# package_default --> use standard values in package `pypmc`
if vb_initialization == "package_default":
    vb = pypmc.mix_adapt.variational.GaussianInference(data, vb_initial_K)
# else --> unrecognized initialization scheme
else:
    raise ValueError("I don't know what you mean by `vb_initialization` = \"%s\"" %vb_initialization)

# run the variational bayes
vb_converged = vb.run(N_max_vb, verbose=True, rel_tol=vb_rel_tol, abs_tol=vb_abs_tol)

# extract gaussian mixture from GaussianInference instance
reduced_proposal = vb.make_mixture()

# cannot trust component weights from Markov chain --> make them uniform
reduced_proposal.weights[:] = 1.
reduced_proposal.normalize()

# save number of components; if and in which step variational bayes converged
params.update( [('final_number_of_components', len(reduced_proposal)), ('vb_converge_step', vb_converged)] )

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
