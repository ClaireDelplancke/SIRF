#! /bin/bash

loc_data=/home/petmr/data/cardiac_resp/pet
loc_algo=~/devel/claire/SIRF/examples/Python/PETMR
loc_image=/home/sirfuser/data/cardiac_resp/pet/recons/ungated_alpha_5.0_pdhg_noprecond_gamma_1.0/recons
                       

python $loc_algo/compute_profile.py                                  \
--outp "ungated"                                                     \
-v 0                                                                 \
-S "$loc_data/STIRungated/sinospan11_f1g1d0b0.hs"                    \
-R "$loc_data/STIRungated/total_background.hs"                       \
-n "$loc_data/NORM.n.hdr"                                          \
-a "$loc_data/MU.nii"                                            \
-t def                                                                 \
--numThreads=27                                                        \
-I "$loc_image/ungated_pdhg_Reg-FGP_TV-alpha5.0_nGates1_nSubsets1_pdhg_noPrecond_gamma1.0_wAC_wNorm_wRands-riters100_noMotion_iters_2000.nii"                          



