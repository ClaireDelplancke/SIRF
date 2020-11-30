#! /bin/bash

loc_data=/home/sirfuser/data/cardiac_resp
loc_algo=~/devel/claire/SIRF/examples/Python/PETMR
loc_reco=~/devel/claire/rebin_notrans_gated_pdhg/recons
loc_param=~/devel/claire/rebin_notrans_gated_pdhg/params
                       

python $loc_algo/PET_MCIR_PD.py             \
-o rebin_notrans_gated_pdhg                   \
--algorithm=pdhg                            \
-r FGP_TV                                   \
--outpath=$loc_reco                        \
--param_path=$loc_param                    \
-e 2000                                     \
--update_obj_fn_interval=10                \
--descriptive_fname                         \
-v 0                                        \
-S "$loc_data/pet/EM_g*.hs"                  \
-R "$loc_data/pet/total_background_g*.hs"      \
-n "$loc_data/pet/NORM.n.hdr"               \
-a "$loc_data/pet/MU_copy*.nii"                  \
--nifti                                   \
--alpha=1.0                                \
--dxdy=3.12117                             \
--nxny=180                                 \
--numThreads=9                            \
--numSegsToCombine=11 --numViewsToCombine=2   



