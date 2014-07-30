Bs_to_ll                                      = {"B^0_s->mu^+mu^-::BR@LHCb-2013D",
                                                 "B^0_s->mu^+mu^-::BR@CMS-2013B"}

B_to_Pll_form_factors                         = {"B->K::f_+@HPQCD-2013A"}

Bplus_to_Pll_large_recoil_no_LHCb             = {"B^+->K^+mu^+mu^-::BR[1.00,6.00]@Belle-2009",
                                                 "B^+->K^+mu^+mu^-::BR[1.00,6.00]@CDF-2012",
                                                 "B^+->K^+mu^+mu^-::BR[1.00,6.00]@BaBar-2012",
                                                 "B^+->K^+mu^+mu^-::A_FB[1.00,6.00]@Belle-2009",
                                                 "B^+->K^+mu^+mu^-::A_FB[1.00,6.00]@CDF-2012"}

Bplus_to_Pll_large_recoil_BR_LHCb2014         = {"B^+->K^+mu^+mu^-::BR[1.10,6.00]@LHCb-2014"}

Bplus_to_Pll_large_recoil_BR_binned_LHCb2014  = {"B^+->K^+mu^+mu^-::BR[1.10,2.00]@LHCb-2014",
                                                 "B^+->K^+mu^+mu^-::BR[2.00,3.00]@LHCb-2014",
                                                 "B^+->K^+mu^+mu^-::BR[3.00,4.00]@LHCb-2014",
                                                 "B^+->K^+mu^+mu^-::BR[4.00,5.00]@LHCb-2014",
                                                 "B^+->K^+mu^+mu^-::BR[5.00,6.00]@LHCb-2014"}

Bplus_to_Pll_large_recoil_AFB_FH_LHCb2014     = {"B^+->K^+mu^+mu^-::A_FB[1.10,6.00]@LHCb-2014",
                                                 "B^+->K^+mu^+mu^-::F_H[1.10,6.00]@LHCb-2014"}

Bplus_to_Pll_large_recoil_no_LHCb_BR          = Bplus_to_Pll_large_recoil_no_LHCb | \
                                                Bplus_to_Pll_large_recoil_AFB_FH_LHCb2014

Bplus_to_Pll_large_recoil                     = Bplus_to_Pll_large_recoil_no_LHCb_BR | \
                                                Bplus_to_Pll_large_recoil_BR_LHCb2014

Bplus_to_Pll_large_recoil_sub_binned          = Bplus_to_Pll_large_recoil_no_LHCb_BR | \
                                                Bplus_to_Pll_large_recoil_BR_binned_LHCb2014

Bplus_to_Pll_low_recoil_no_LHCb               = {"B^+->K^+mu^+mu^-::BR[14.18,16.00]@Belle-2009",
                                                 "B^+->K^+mu^+mu^-::BR[16.00,22.86]@Belle-2009",
                                                 "B^+->K^+mu^+mu^-::BR[16.00,22.86]@BaBar-2012",
                                                 "B^+->K^+mu^+mu^-::BR[14.21,16.00]@BaBar-2012",
                                                 "B^+->K^+mu^+mu^-::BR[14.18,16.00]@CDF-2012",
                                                 "B^+->K^+mu^+mu^-::BR[16.00,22.86]@CDF-2012",
                                                 "B^+->K^+mu^+mu^-::A_FB[14.18,16.00]@Belle-2009",
                                                 "B^+->K^+mu^+mu^-::A_FB[16.00,22.86]@Belle-2009",
                                                 "B^+->K^+mu^+mu^-::A_FB[14.18,16.00]@CDF-2012",
                                                 "B^+->K^+mu^+mu^-::A_FB[16.00,22.86]@CDF-2012"}

Bplus_to_Pll_low_recoil_LHCb2014              = {"B^+->K^+mu^+mu^-::BR[15.00,22.00]@LHCb-2014",
                                                 "B^+->K^+mu^+mu^-::A_FB[15.00,22.00]@LHCb-2014",
                                                 "B^+->K^+mu^+mu^-::F_H[15.00,22.00]@LHCb-2014"}

Bplus_to_Pll_low_recoil                       = Bplus_to_Pll_low_recoil_no_LHCb | \
                                                Bplus_to_Pll_low_recoil_LHCb2014
