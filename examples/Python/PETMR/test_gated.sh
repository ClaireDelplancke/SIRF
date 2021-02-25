#! /bin/bash

loc_data=/home/sirfuser/data/cardiac_resp
loc_algo=~/devel/claire/SIRF/examples/Python/PETMR
loc_reco=~/devel/claire/cluster_test/recons
loc_param=~/devel/claire/cluster_test/params
                       

python $loc_algo/PET_MCIR_PD.py             \
-o gated_test_rescaled                      \
--algorithm=pdhg                            \
-r FGP_TV                                   \
--outpath=$loc_reco                        \
--param_path=$loc_param                    \
-e 10                                        \
--update_obj_fn_interval=2                  \
--descriptive_fname                         \
-v 0                                        \
-S "$loc_data/pet/EM_g*.hs"                  \
-R "$loc_data/pet/total_background_g*.hs"      \
-n "$loc_data/pet/NORM.n.hdr"               \
-a "$loc_data/pet/MU_g*.nii"                  \
-T "$loc_data/pet/transf_g*.nii"            \
-t def                                    \
--nifti                                   \
--alpha=5.0                                \
--dxdy=3.12117                             \
--nxny=180                                 \
--numThreads=27