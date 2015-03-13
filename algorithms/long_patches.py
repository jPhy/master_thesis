import numpy as np
import pypmc
from pypmc.density.mixture import create_gaussian_mixture

def partition(N, k):
    '''Distributre ``N`` into ``k`` partitions such that each partition
    takes the value ``N//k`` or ``N//k + 1`` where ``//`` denotes integer
    division.

    Example: N = 5, k = 2  -->  return [3, 2]

    '''
    out = [N // k] * k
    remainder = N % k
    for i in range(remainder):
        out[i] += 1
    return out

def make_long_patch_gaussian_mixture(data, K_g=15, critical_r=2.):
    '''Use samples from Markov-chains to form a Gaussian Mixture (to be
    used as initial guess for VB). This is done using "long patches" as
    in [Allen Fred].

    :param data:

        Iterable of vector-like arrays; the individual items are interpreted
        as points from an individual chain.

    :param K_g:

        Integer; the number of components per chain group.

    :param critical_r:

        Float; the maximum R value a chain group may have.

    '''
    def append_components(means, covs, data, partition):
        subdata_start = 0
        subdata_stop  = partition[0]
        for len_subdata in partition:
            subdata = data[subdata_start:subdata_stop]
            means.append( np.mean(subdata,   axis=0) )
            covs.append ( np.cov (subdata, rowvar=0) )
            subdata_start += len_subdata
            subdata_stop  += len_subdata


    chain_groups = pypmc.mix_adapt.r_value.r_group([np.mean(chain_values[:], axis=0) for chain_values in data],
                                                   [np.cov(chain_values[:], rowvar=0) for chain_values in data],
                                                   len(data[0]), critical_r)

    print 'found %i chain grous\n' %len(chain_groups)

    long_patches_means = []
    long_patches_covs = []
    for group in chain_groups:
        # we want K_g components from k_g = len(group) chains
        k_g = len(group)
        if K_g >= k_g:
            # find minimal lexicographic integer partition
            n = partition(K_g, k_g)
            for i, chain_index in enumerate(group):
                # need to partition in n[i] parts
                data_full_chain = data[chain_index]
                # find minimal lexicographic integer partition of chain_length into n[i]
                this_patch_lengths = partition(len(data_full_chain), n[i])
                append_components(long_patches_means, long_patches_covs, data_full_chain, this_patch_lengths)
        else:
            # form one long chain and set k_g = 1
            k_g = 1
            # make one large chain
            data_full_chain = np.vstack([data[i] for i in group])
            # need to partition into K_g parts -- > minimal lexicographic integer partition
            this_patch_lengths = partition(len(data_full_chain), K_g)
            append_components(long_patches_means, long_patches_covs, data_full_chain, this_patch_lengths)

    return create_gaussian_mixture(long_patches_means, long_patches_covs)
