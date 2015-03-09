from pypmc.tools import plot_mixture
import matplotlib.pyplot as plt

# reprogram vb.run with plot output
def vb_update_movie(vb, iterations=1000, prune=1., rel_tol=1e-10, abs_tol=1e-5, verbose=False):
        r'''Run variational-Bayes parameter updates and check for convergence
        using the change of the log likelihood bound of the current and the last
        step. Convergence is not declared if the number of components changed,
        or if the bound decreased. For the standard algorithm, the bound must
        increase, but for modifications, this useful property may not hold for
        all parameter values.

        Return the number of iterations at convergence, or None.

        :param iterations:
            Maximum number of updates.

        :param prune:
            Call :py:meth:`prune` after each update; i.e., remove components
            whose associated effective number of samples is below the
            threshold. Set `prune=0` to deactivate.
            Default: 1 (effective samples).

        :param rel_tol:
            Relative tolerance :math:`\epsilon`. If two consecutive values of
            the log likelihood bound, :math:`L_t, L_{t-1}`, are close, declare
            convergence. More precisely, check that

            .. math::
                \left\| \frac{L_t - L_{t-1}}{L_t} \right\| < \epsilon .

        :param abs_tol:
            Absolute tolerance :math:`\epsilon_{a}`. If the current bound
            :math:`L_t` is close to zero, (:math:`L_t < \epsilon_{a}`), declare
            convergence if

            .. math::
                \| L_t - L_{t-1} \| < \epsilon_a .

        :param verbose:
            Output status information after each update.

        '''
        old_K = None
        for i in range(1, iterations + 1):
            # recompute bound in 1st step or if components were removed
            if vb.K == old_K:
                old_bound = bound
            else:
                old_bound = vb.likelihood_bound()
                if verbose:
                    print('New bound=%g, K=%d, N_k=%s' % (old_bound, vb.K, vb.N_comp))

            vb.update()
            bound = vb.likelihood_bound()

            if verbose:
                print('After update %d: bound=%.15g, K=%d, N_k=%s' % (i, bound, vb.K, vb.N_comp))

            if bound < old_bound:
                print('WARNING: bound decreased from %g to %g' % (old_bound, bound))

            # plot importance samples every 10th iteration
            if not i % 10:
                plt.clf()
                plt.title('Proposal density - iteration %i' %i)
                plot_mixture(vb.make_mixture(), 1, 2, visualize_weights=True, cmap='jet')
                plt.colorbar()
                plt.xlabel('$x_2$')
                plt.ylabel('$x_3$')
                plt.xlim(-30,10)
                plt.ylim(-30,30)
                plt.draw()

             # exact convergence
            if bound == old_bound:
                return i
            # approximate convergence
            # but only if bound increased
            diff = bound - old_bound
            if diff > 0:
                # handle case when bound is close to 0
                if abs(bound) < abs_tol:
                    if abs(diff) < abs_tol:
                        return i
                else:
                    if abs(diff / bound) < rel_tol:
                        return i

            # save K *before* pruning
            old_K = vb.K
            vb.prune(prune)
        # not converged
        return None
