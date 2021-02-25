#! /bin/bash

loc_data=/home/petmr/data/cardiac_resp
loc_algo=~/devel/claire/SIRF/examples/Python/PETMR
loc_image=/home/sirfuser/data/cardiac_resp/pet/recons/gated_alpha_5.0_pdhg_noprecond_gamma_1.0/recons
                       

python $loc_algo/compute_profile.py            \
--outp "gated8"                                \
-v 0                                           \
-S "$loc_data/pet/EM_g*.hs"                    \
-R "$loc_data/pet/total_background_g*.hs"      \
-n "$loc_data/pet/NORM.n.hdr"                  \
-a "$loc_data/pet/MU_g*.nii"                   \
-T "$loc_data/pet/transf_g*.nii"               \
-t def                                         \
--numThreads=27                                \
-I "$loc_image/gated_pdhg_Reg-FGP_TV-alpha5.0_nGates8_nSubsets1_pdhg_noPrecond_gamma1.0_wAC_wNorm_wRands-riters100_iters_500.nii"                          



