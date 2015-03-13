# for a new algorithm only change the part which is marked as such

from __future__ import division
import numpy as np

# need load_save from '..'
import sys; sys.path.append('..')
from load_save import save_final_proposal, load_first_proposal

# load parameters
from final_run_params import params
locals().update(params)

# load proposal from mcmc
if type(first_proposal_index) is int:
    first_proposal = load_first_proposal(first_proposal_index)
elif type(first_proposal_index) is str:
    if first_proposal_index == 'good_original':
        if dim == 2:
            first_proposal = load_first_proposal(0)
        elif dim == 10:
            first_proposal = load_first_proposal(65)
        elif dim == 20:
            first_proposal = load_first_proposal(28)
        else:
            raise NotImplementedError('No such first proposal')

    elif first_proposal_index == 'bad_original':
        if dim == 10:
            first_proposal = load_first_proposal(52)
        elif dim == 20:
            first_proposal = load_first_proposal(23)
        else:
            raise NotImplementedError('No such first proposal')

    elif first_proposal_index == 'good_variational':
        if dim == 2:
            first_proposal = load_first_proposal(7)
        elif dim == 10:
            first_proposal = load_first_proposal(58)
        elif dim == 20:
            first_proposal = load_first_proposal(35)
        else:
            raise NotImplementedError('No such first proposal')

    elif first_proposal_index == 'bad_variational':
        if dim == 2:
            first_proposal = load_first_proposal(5)
        elif dim == 10:
            first_proposal = load_first_proposal(43)
        elif dim == 20:
            first_proposal = load_first_proposal(37)
        else:
            raise NotImplementedError('No such first proposal')

    else:
        raise ValueError("I don't know what you mean by `first_proposal_index` = \"%s\"" %first_proposal_index)
else:
    raise TypeError('``first_proposal_index`` must be string or integer')
assert dim == first_proposal.creation_parameters['dim'], "``first_proposal`` has wrong dimension"

# save first_proposal's paramters
params.update( [('first_proposal_params', first_proposal.creation_parameters)] )

# load the target function
from target import LogTarget; log_target = LogTarget(dim)

# load pypmc
import pypmc
from pypmc.tools.convergence import perp, ess

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

def live_components(proposal):
    "Return the number of non-zero weighted components of a mixture Density"
    return (proposal.weights!=0).sum()


if method == "original":

    sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, first_proposal)
    previous_perp = - np.inf
    converge_step = None

    for i in range(max_sampling_steps):
        print "step", i

        # run len(sampler.proposal) * N_c steps
        latent = sampler.run(N_c * len(sampler.proposal), trace_sort=True)

        # get the weighted samples that have just been generated
        weighted_samples = sampler.history[-1]

        # calculate perplexity
        this_perp = pypmc.tools.convergence.perp(weighted_samples[:,0])

        # stop if convergence criterion reached
        if i >= min_steps and (this_perp - previous_perp) / this_perp < .05:
            converge_step = i
            print 'PMC converged in step ' + str(i)
            break
        else:
            previous_perp = this_perp

        # update the proposal using PMC
        pypmc.mix_adapt.pmc.gaussian_pmc(weighted_samples[:,1:], sampler.proposal, weights=weighted_samples[:,0],
                                         latent=latent, rb=rb, mincount=mincount, copy=False)

        # can prune here because samples are not reused
        sampler.proposal.prune()

        print 'have %i live components PMC step %i\n' %(live_components(sampler.proposal), i)

    params.update( [('final_number_of_components', live_components(sampler.proposal)), ('converge_step', converge_step)] )

elif method == 'PMC_pripos':

    sampler = pypmc.sampler.importance_sampling.DeterministicIS(log_target, first_proposal, std_weights=True)
    previous_perp = - np.inf
    latent = []
    converge_step = None

    for i in range(max_sampling_steps):
        print "step", i

        # run len(sampler.proposal) * N_c steps
        latent.extend(sampler.run(N_c * live_components(sampler.proposal), trace_sort=True))

        # get ALL weighted samples
        weighted_samples = sampler.history[:]

        # calculate perplexity using the latest run only
        this_perp = pypmc.tools.convergence.perp(sampler.std_weights[-1][:,0])

        # stop if convergence criterion reached
        if i >= min_steps and (this_perp - previous_perp) / this_perp < .05:
            converge_step = i
            print 'PMC converged in step ' + str(i)
            break
        else:
            previous_perp = this_perp

        # update the proposal using PMC
        pypmc.mix_adapt.pmc.gaussian_pmc(weighted_samples[:,1:], sampler.proposal, weights=weighted_samples[:,0],
                                         latent=latent, rb=rb, mincount=mincount, copy=False)

        # can NOT prune here because samples are reused

        print 'have %i live components PMC step %i\n' %(live_components(sampler.proposal), i)

    params.update( [('final_number_of_components', live_components(sampler.proposal)), ('converge_step', converge_step)] )

elif method == 'VB_pripos':

    sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, first_proposal)
    previous_perp = - np.inf
    converge_step = None

    for i in range(max_sampling_steps):
        print "step", i

        # run len(sampler.proposal) * N_c steps
        sampler.run(N_c * len(sampler.proposal), trace_sort=True)

        # get the weighted samples that have just been generated
        weighted_samples = sampler.history[-1]

        # calculate perplexity
        this_perp = pypmc.tools.convergence.perp(weighted_samples[:,0])

        # stop if convergence criterion reached
        if i >= min_steps and (this_perp - previous_perp) / this_perp < .05:
            converge_step = i
            print 'converged in step ' + str(i)
            break
        else:
            previous_perp = this_perp

        # update the proposal using VB
        if i == 0:
            if first_VB_prior:
                # cannot rely on alpha0
                cleaned_prior = first_proposal.creation_parameters['vb_posterior2prior'].copy()
                cleaned_prior.pop('alpha0')
                vb = pypmc.mix_adapt.variational.GaussianInference(weighted_samples[:,1:], initial_guess=sampler.proposal,
                                                                   weights=weighted_samples[:,0], **cleaned_prior)
            else:
                vb = pypmc.mix_adapt.variational.GaussianInference(weighted_samples[:,1:], initial_guess=sampler.proposal,
                                                                   weights=weighted_samples[:,0])
        else:
            print vb.posterior2prior()
            vb = pypmc.mix_adapt.variational.GaussianInference(weighted_samples[:,1:], initial_guess=sampler.proposal,
                                                               weights=weighted_samples[:,0], **vb.posterior2prior())
        vb.run(max_vb_steps, abs_tol=10**-5, rel_tol=10**-10, prune=mincount, verbose=True)
        sampler.proposal = vb.make_mixture()

        print 'have %i live components in step %i\n' %(live_components(sampler.proposal), i)

    params.update( [('final_number_of_components', live_components(sampler.proposal)), ('converge_step', converge_step)] )

elif method == 'VB_cornuet':

    sampler = pypmc.sampler.importance_sampling.DeterministicIS(log_target, first_proposal, std_weights=True)
    previous_perp = - np.inf
    converge_step = None

    for i in range(max_sampling_steps):
        print "step", i

        # run len(sampler.proposal) * N_c steps
        sampler.run(N_c * live_components(sampler.proposal), trace_sort=True)

        # get ALL weighted samples
        weighted_samples = sampler.history[:]

        # calculate perplexity using the latest run only
        this_perp = pypmc.tools.convergence.perp(sampler.std_weights[-1][:,0])

        # stop if convergence criterion reached
        if i >= min_steps and (this_perp - previous_perp) / this_perp < .05:
            converge_step = i
            print 'converged in step ' + str(i)
            break
        else:
            previous_perp = this_perp

        # update the proposal using VB
        if i == 0:
            if first_VB_prior:
                # cannot rely on alpha0
                cleaned_prior = first_proposal.creation_parameters['vb_posterior2prior'].copy()
                cleaned_prior.pop('alpha0')
                vb = pypmc.mix_adapt.variational.GaussianInference(weighted_samples[:,1:], initial_guess=sampler.proposal,
                                                                   weights=weighted_samples[:,0], **cleaned_prior)
            else:
                vb = pypmc.mix_adapt.variational.GaussianInference(weighted_samples[:,1:], initial_guess=sampler.proposal,
                                                                   weights=weighted_samples[:,0])
        else:
            if first_VB_prior:
                vb = pypmc.mix_adapt.variational.GaussianInference(weighted_samples[:,1:], initial_guess=sampler.proposal,
                                                               weights=weighted_samples[:,0], **cleaned_prior) #no posterior2prior here!
            else:
                vb = pypmc.mix_adapt.variational.GaussianInference(weighted_samples[:,1:], initial_guess=sampler.proposal,
                                                               weights=weighted_samples[:,0])
        vb.run(max_vb_steps, abs_tol=10**-5, rel_tol=10**-10, prune=mincount, verbose=True)
        sampler.proposal = vb.make_mixture()

        print 'have %i live components in step %i\n' %(live_components(sampler.proposal), i)

    params.update( [('final_number_of_components', live_components(sampler.proposal)), ('converge_step', converge_step)] )

else:
    raise ValueError("I don't know what you mean by `method` = \"%s\"" %method)

final_proposal = sampler.proposal

# ----------------------- replace algorithm above -----------------------

# ***********************************************************************
# ****************** nothing below should be changed ! ******************
# ***********************************************************************





# use importance sampling to calculate perplexity and effective sample size

# define an ImportanceSampler object using ``reduced_proposal``
sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, final_proposal)

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
save_final_proposal(final_proposal, params)
