import eos
from copy import deepcopy

def get_nuisance(constraints):
    copied_constraints = deepcopy(constraints)
    out = set(CKM_all + quark_masses)

    if "B->K::f_+@HPQCD-2013A" in copied_constraints:
        copied_constraints.remove("B->K::f_+@HPQCD-2013A")
        out.update(form_factors)

    for constraint in constraints:

        if constraint.startswith("B^0_s->mu^+mu^-"):
            copied_constraints.remove(constraint)
            out.update(decay_constants)

        if constraint.startswith("B^+->K^+mu^+mu^-"):
            copied_constraints.remove(constraint)
            out.update(form_factors + subleading_B_to_Pll)

    if copied_constraints:
        raise ValueError("Unparsable constraints: " + str(constraints) )

    return list(out)



# defines the allowed parameter range in terms of sigmas for Gaussian and LogGamma distributions
N_SIGMAS = 3


# CKM parameters UTfit Summer 2013 (post-EPS13); Tree level Fit
# -------------------------------------------------------------
c = 0.806; sigma_u = sigma_l = 0.020
CKM_A      = eos.LogPrior.Gauss("CKM::A"     , range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                               lower=c-sigma_l, central=c , upper=c+sigma_u)

c = 0.2253; sigma_u = sigma_l = 0.0006
CKM_lambda = eos.LogPrior.Gauss("CKM::lambda", range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                               lower=c-sigma_l, central=c , upper=c+sigma_u)

c = 0.132; sigma_u = sigma_l = 0.049
CKM_rhobar = eos.LogPrior.Gauss("CKM::rhobar", range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                               lower=c-sigma_l, central=c , upper=c+sigma_u)

c = 0.369; sigma_u = sigma_l = 0.050
CKM_etabar = eos.LogPrior.Gauss("CKM::etabar", range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                               lower=c-sigma_l, central=c , upper=c+sigma_u)

CKM_all = [CKM_A, CKM_lambda, CKM_rhobar, CKM_etabar]


# quark masses from PDG 2012 (on 07/23/2014 PDG2014 was not available yet)
# ------------------------------------------------------------------------
c = 1.275; sigma_u = sigma_l = 0.025
m_c        = eos.LogPrior.Gauss("mass::c"       , range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                                  lower=c-sigma_l, central=c , upper=c+sigma_u)

c = 4.18; sigma_u = sigma_l = 0.03
m_b_MS_bar = eos.LogPrior.Gauss("mass::b(MSbar)", range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                                  lower=c-sigma_l, central=c , upper=c+sigma_u)

quark_masses = [m_c, m_b_MS_bar]

# decay-constant::B_s from Lattice Averages, June 2013
# -------------------
c = 0.2276; sigma_u = sigma_l = 0.005
decay_B_s = eos.LogPrior.Gauss("decay-constant::B_s" , range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                                       lower=c-sigma_l, central=c , upper=c+sigma_u)

decay_constants = [decay_B_s]

# form factors
# ------------
c = 0.34; sigma_u = sigma_l = 0.05
F_0 = eos.LogPrior.Gauss   ("B->K::F^p(0)@KMPW2010", range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                                     lower=c-sigma_l, central=c , upper=c+sigma_u)

c = -2.1; sigma_u = 0.9; sigma_l = 1.6
b_1 = eos.LogPrior.LogGamma("B->K::b^p_1@KMPW2010" , range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                                     lower=c-sigma_l, central=c , upper=c+sigma_u)

form_factors = [F_0, b_1]


# subleading B->Pll
# -----------------
c = 0.0; sigma_u = sigma_l = 0.15
lambda_pseudo_low_recoil = eos.LogPrior.Gauss("B->Pll::Lambda_pseudo@LowRecoil"    , range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                                                                     lower=c-sigma_l, central=c , upper=c+sigma_u)

c = 0.0; sigma_u = sigma_l = 0.50
lambda_pseudo_large_recoil = eos.LogPrior.Gauss("B->Pll::Lambda_pseudo@LargeRecoil", range_min=c-N_SIGMAS*sigma_l, range_max=c+N_SIGMAS*sigma_u,
                                                                                     lower=c-sigma_l, central=c , upper=c+sigma_u)

subleading_B_to_Pll = [lambda_pseudo_low_recoil, lambda_pseudo_large_recoil]
