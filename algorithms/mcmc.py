from __future__ import division
import numpy as np
import pypmc

def create_mcmc_samples(log_target, k=10, N_burn_in=10**3, N_MCMC=2*10**4, N_adapt=500, scale=.1, upper=3., lower=-3.):
    """Run ``k`` markov-chains for ``N_burn_in`` steps, discard
    and run again for ``N_MCMC`` steps whereby the proposals are
    to be adapted every ``N_adapt`` steps.
    Proposal covariances are initialized as np.eye(dim)*scale.
    Starting values for the chains are randomly chosen within the
    hypercube between ``upper`` and ``lower``.
    Dimension is read from ``log_target.dim``.

    """
    dim = log_target.dim

    # define the initial proposal for the local random walk markov chain
    prop_sigma = scale * np.eye(dim)
    prop = pypmc.density.gauss.LocalGauss(prop_sigma)

    # choose random initializations to run k chains
    starts = []
    for i in range(k):
        point = np.random.rand(dim)*(upper-lower) + lower
        while np.isinf(log_target(point)):
            point = np.random.rand(dim)*(upper-lower) + lower
        starts.append(point)

    # define the markov chain objects
    mcs = [pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, prop, starts[i]) for i in range(k)]


    # run and discard burn-in
    for i, mc in enumerate(mcs):
        print "burn in chain", i
        mc.run(N_burn_in)
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
    accept_rates = np.array([float(accept_counts[i]) / len(values[i]) for i in range(k)])
    for i, count in enumerate(accept_counts):
        print "Chain %i accepted %4.2f%% of the proposed points" % (i, accept_rates[i] * 100)

    # check optimal range for accep_rate
    assert ( accept_rates > .15 ).all()
    assert ( accept_rates < .35 ).all()

    # return the samples
    return values
