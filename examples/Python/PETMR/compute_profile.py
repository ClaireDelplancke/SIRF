# -*- coding: utf-8 -*-

"""Compute profile of synthetic acq data and compare with real acq data

Usage:
  PET_MCIR_PD [--help | options]

Options:
  -I <str>, --image=<str>           Image to forward project
  -T <pattern>, --trans=<pattern>   transformation pattern, * or % wildcard
                                    (e.g., tm_ms*.txt). Enclose in quotations.
  -t <str>, --trans_type=<str>      transformation type (tm, disp, def)
                                    [default: tm]
  -S <pattern>, --sino=<pattern>    sinogram pattern, * or % wildcard
                                    (e.g., sino_ms*.hs). Enclose in quotations.
  -a <pattern>, --attn=<pattern>    attenuation pattern, * or % wildcard
                                    (e.g., attn_ms*.hv). Enclose in quotations.
  -R <pattern>, --rand=<pattern>    randoms pattern, * or % wildcard
                                    (e.g., rand_ms*.hs). Enclose in quotations.
  -n <norm>, --norm=<norm>          ECAT8 bin normalization file
  -v <int>, --verbosity=<int>       STIR verbosity [default: 0]   
  --numSegsToCombine=<val>          Rebin all sinograms, with a given number of
                                    segments to combine. Increases speed.
  --numViewsToCombine=<val>         Rebin all sinograms, with a given number of
                                    views to combine. Increases speed.
  --numThreads=<int>                Number of threads to use
  --StorageSchemeMemory             Use memory storage scheme
  --gpu                             use GPU projector
  --outp=<str>                      output file prefix 
  --nSino=<int>                     sinogram number [default: 400] 
  --nLine=<int>                     line number [default: 150]
  --MeanLength=<int>                number of lines over which to average [default: 10]
"""

# SyneRBI Synergistic Image Reconstruction Framework (SIRF)
# Copyright 2020 University College London.
#
# This is software developed for the Collaborative Computational
# Project in Synergistic Reconstruction for Biomedical Imaging
# (formerly CCP PETMR)
# (http://www.ccpsynerbi.ac.uk/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from functools import partial
from os import path
import os
from glob import glob
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np

from sirf.Utilities import error, show_2D_array, examples_data_path
import sirf.Reg as reg
import sirf.STIR as pet
from cil.framework import BlockDataContainer, ImageGeometry, BlockGeometry
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import \
    KullbackLeibler, BlockFunction, IndicatorBox, MixedL21Norm, ScaledFunction, L2NormSquared
from cil.optimisation.operators import \
    CompositionOperator, BlockOperator, LinearOperator, GradientOperator, ScaledOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from ccpi.filters import regularisers
from cil.utilities.multiprocessing import NUM_THREADS

__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)

###########################################################################
# Global set-up
###########################################################################

# storage scheme
if args['--StorageSchemeMemory']:
    pet.AcquisitionData.set_storage_scheme('memory')
else:
    pet.AcquisitionData.set_storage_scheme('default')
# Verbosity
pet.set_verbosity(int(args['--verbosity']))
if int(args['--verbosity']) == 0:
    msg_red = pet.MessageRedirector(None, None, None)
# Number of threads
numThreads = args['--numThreads'] if args['--numThreads'] else NUM_THREADS
pet.set_max_omp_threads(numThreads)



def main():
    """Run main function."""

    use_gpu = True if args['--gpu'] else False

    ###########################################################################
    # Parse input files
    ###########################################################################

    [num_ms, trans_files, sino_files, attn_files, rand_files] = \
        get_filenames(args['--trans'],args['--sino'],args['--attn'],args['--rand'])

    ###########################################################################
    # Read input
    ###########################################################################

    [trans, sinos_raw, attns, rands_raw] = \
        read_files(trans_files, sino_files, attn_files, rand_files, args['--trans_type'])

    sinos = pre_process_sinos(sinos_raw, num_ms)
    rands = pre_process_sinos(rands_raw, num_ms)

    ###########################################################################
    # Read in recon image
    ###########################################################################

    image = pet.ImageData(args['--image'])

    ###########################################################################
    # Set up resamplers
    ###########################################################################

    if trans is None:
        resamplers = None
    else:
        resamplers = [get_resampler(image, trans=tran) for tran in trans]

    ###########################################################################
    # Resample attenuation images (if necessary)
    ###########################################################################

    resampled_attns = resample_attn_images(num_ms, attns, trans, use_gpu, image)
    print ("resampled_attns", len (resampled_attns))

    ###########################################################################
    # Set up acquisition models (one per motion state)
    ###########################################################################

    acq_models, masks = set_up_acq_models(
        num_ms, sinos, rands, resampled_attns, image, use_gpu)

    ###########################################################################
    # Forward project the image (one projection per motion state)
    ###########################################################################

    synth_sinos = [num_ms * Ki.direct(image) + randsi 
                        for Ki, randsi in zip(acq_models, rands)]

    ###########################################################################
    # Save images of synthetic and real data
    ###########################################################################

    nSino = int(args['--nSino'])
    
    for i in range(num_ms):
        sino_norm = sinos[i].norm()
        plt.subplot(1,3,1)
        plt.imshow(sinos[i].as_array()[0, nSino, :, :])
        plt.colorbar()
        plt.gca().set_title('Real')

        plt.subplot(1,3,2)
        plt.imshow(synth_sinos[i].as_array()[0, nSino, :, :])
        plt.colorbar()
        plt.gca().set_title('Synth')

        plt.subplot(1,3,3)
        plt.imshow((sinos[i]-synth_sinos[i]).as_array()[0, nSino, :, :])
        plt.colorbar()
        plt.gca().set_title('Diff')

        plt.suptitle('Gate {},'.format(i) + 'Real data, L2norm = {} \n'.format(sino_norm)
                        + 'Synth data, L2norm = {} \n'.format(synth_sinos[i].norm()) 
                        + 'Relative dist {}'.format((sinos[i]-synth_sinos[i]).norm()/sino_norm))

        plt.savefig( '{}_sinos_gate{}.png'.format(str(args['--outp']),i) )
        plt.close()
    
    ###########################################################################
    # Compute profiles
    ###########################################################################  
     
    N = int(args['--MeanLength'])
    nLine = int(args['--nLine'])

    for i in range(num_ms):
        sino_a = sinos[i].as_array()
        sy_sino_a = synth_sinos[i].as_array()
        mean_line = 1/N * sum([sino_a[0, nSino, nLine+j, :] for j in range(N)]) 
        sy_mean_line = 1/N * sum([sy_sino_a[0, nSino, nLine+j, :] for j in range(N)]) 

        plt.plot(mean_line)
        plt.plot(sy_mean_line)
        plt.legend(['real data', 'synth data'])
        plt.suptitle('Gate {}, nSino {}, nLine {}'.format(i, nSino, nLine))

        plt.savefig( '{}_lines_gate{}.png'.format(str(args['--outp']),i) )
        plt.close()




def get_filenames(trans, sino, attn, rand):
    """Get filenames."""
    trans_pattern = str(trans).replace('%', '*')
    sino_pattern = str(sino).replace('%', '*')
    attn_pattern = str(attn).replace('%', '*')
    rand_pattern = str(rand).replace('%', '*')    
    if sino_pattern is None:
        raise AssertionError("--sino missing")
    trans_files = sorted(glob(trans_pattern))
    sino_files = sorted(glob(sino_pattern))
    attn_files = sorted(glob(attn_pattern))
    rand_files = sorted(glob(rand_pattern))

    num_ms = len(sino_files)
    # Check some sinograms found
    if num_ms == 0:
        raise AssertionError("No sinograms found at {}!".format(sino_pattern))
    # Should have as many trans as sinos
    if len(trans_files) > 0 and num_ms != len(trans_files):
        raise AssertionError("#trans should match #sinos. "
                             "#sinos = " + str(num_ms) +
                             ", #trans = " + str(len(trans_files)))
    # If any rand, check num == num_ms
    if len(rand_files) > 0 and len(rand_files) != num_ms:
        raise AssertionError("#rand should match #sinos. "
                             "#sinos = " + str(num_ms) +
                             ", #rand = " + str(len(rand_files)))

    # For attn, there should be 0, 1 or num_ms images
    if len(attn_files) > 1 and len(attn_files) != num_ms:
        raise AssertionError("#attn should be 0, 1 or #sinos")

    return [num_ms, trans_files, sino_files, attn_files, rand_files]



def read_files(trans_files, sino_files, attn_files, rand_files, trans_type):
    """Read files."""
    if not trans_files:
        trans = None
    else:
        if trans_type == "tm":
            trans = [reg.AffineTransformation(file) for file in trans_files]
        elif trans_type == "disp":
            trans = [reg.NiftiImageData3DDisplacement(file)
                     for file in trans_files]
        elif trans_type == "def":
            trans = [reg.NiftiImageData3DDeformation(file)
                     for file in trans_files]
        else:
            raise error("Unknown transformation type")

    sinos_raw = [pet.AcquisitionData(file) for file in sino_files]
    attns = [pet.ImageData(file) for file in attn_files]
    
    # fix a problem with the header which doesn't allow
    # to do algebra with randoms and sinogram
    rands_arr = [pet.AcquisitionData(file).as_array() for file in rand_files]
    rands_raw = [ s * 0 for s in sinos_raw ]
    for r,a in zip(rands_raw, rands_arr):
        r.fill(a)
    
    return [trans, sinos_raw, attns, rands_raw]




def pre_process_sinos(sinos_raw, num_ms):
    """Preprocess raw sinograms.

    Make positive if necessary and do any required rebinning."""
    # If empty (e.g., no randoms), return
    if not sinos_raw:
        return sinos_raw
    # Loop over all sinograms
    sinos = [0]*num_ms
    for ind in range(num_ms):
        # If any sinograms contain negative values
        # (shouldn't be the case), set them to 0
        sino_arr = sinos_raw[ind].as_array()
        if (sino_arr < 0).any():
            print("Input sinogram " + str(ind) +
                  " contains -ve elements. Setting to 0...")
            sinos[ind] = sinos_raw[ind].clone()
            sino_arr[sino_arr < 0] = 0
            sinos[ind].fill(sino_arr)
        else:
            sinos[ind] = sinos_raw[ind]
        # If rebinning is desired
        segs_to_combine = 1
        if args['--numSegsToCombine']:
            segs_to_combine = int(args['--numSegsToCombine'])
        views_to_combine = 1
        if args['--numViewsToCombine']:
            views_to_combine = int(args['--numViewsToCombine'])
        if segs_to_combine * views_to_combine > 1:
            sinos[ind] = sinos[ind].rebin(segs_to_combine, views_to_combine,do_normalisation=False)
            # only print first time
            if ind == 0:
                print("Rebinned sino dimensions: {sinos[ind].dimensions()}")

    return sinos






def get_resampler(image, ref=None, trans=None):
    """Return a NiftyResample object for the specified transform and image."""
    if ref is None:
        ref = image
    resampler = reg.NiftyResample()
    resampler.set_reference_image(ref)
    resampler.set_floating_image(image)
    resampler.set_padding_value(0)
    resampler.set_interpolation_type_to_linear()
    if trans is not None:
        resampler.add_transformation(trans)
    return resampler


def resample_attn_images(num_ms, attns, trans, use_gpu, image):
    """Resample attenuation images if necessary."""
    resampled_attns = None
    if trans is None:
        resampled_attns = attns
    else:
        if len(attns) > 0:
            resampled_attns = [0]*num_ms
            # if using GPU, dimensions of attn and recon images have to match
            ref = image if use_gpu else None
            for i in range(num_ms):
                # if we only have 1 attn image, then we need to resample into
                # space of each gate. However, if we have num_ms attn images,
                # then assume they are already in the correct position, so use
                # None as transformation.
                tran = trans[i] if len(attns) == 1 else None
                # If only 1 attn image, then resample that. If we have num_ms
                # attn images, then use each attn image of each frame.
                attn = attns[0] if len(attns) == 1 else attns[i]
                resam = get_resampler(attn, ref=ref, trans=tran)
                resampled_attns[i] = resam.forward(attn)
    return resampled_attns

def set_up_acq_models(num_ms, sinos, rands, resampled_attns, image, use_gpu):
    """Set up acquisition models."""
    print("Setting up acquisition models...")

    # From the arguments
    algo = 'pdhg'
    nsub = 1
    norm_file = args['--norm']
    verbosity = int(args['--verbosity'])
  

    if not use_gpu:
        acq_models = [pet.AcquisitionModelUsingRayTracingMatrix() for k in range(nsub * num_ms)]
    else:
        acq_models = [pet.AcquisitionModelUsingNiftyPET() for k in range(nsub * num_ms)]
        for acq_model in acq_models:
            acq_model.set_use_truncation(True)
            acq_model.set_cuda_verbosity(verbosity)
            acq_model.set_num_tangential_LORs(10)

    # create masks
    im_one = image.clone().allocate(1.)
    masks = []



    # If present, create ASM from ECAT8 normalisation data
    asm_norm = None
    if norm_file:
        if not path.isfile(norm_file):
            raise error("Norm file not found: " + norm_file)
        asm_norm = pet.AcquisitionSensitivityModel(norm_file)

    # Loop over each motion state
    for ind in range(num_ms):
        # Create attn ASM if necessary
        asm_attn = None
        if resampled_attns:
            s = sinos[ind]
            ra = resampled_attns[ind]
            am = pet.AcquisitionModelUsingRayTracingMatrix()
            asm_attn = get_asm_attn(s,ra,am)

        # Get ASM dependent on attn and/or norm
        asm = None
        if asm_norm and asm_attn:
            if ind == 0:
                print("ASM contains norm and attenuation...")
            asm = pet.AcquisitionSensitivityModel(asm_norm, asm_attn)
        elif asm_norm:
            if ind == 0:
                print("ASM contains norm...")
            asm = asm_norm
        elif asm_attn:
            if ind == 0:
                print("ASM contains attenuation...")
            asm = asm_attn
                
        # Loop over physical subsets
        for k in range(nsub):
            current = k * num_ms + ind

            if asm:
                acq_models[current].set_acquisition_sensitivity(asm)
            #KT we'll set the background in the KL function below
            #KTif len(rands) > 0:
            #KT    acq_models[ind].set_background_term(rands[ind])

            # Set up
            acq_models[current].set_up(sinos[ind], image)    
            acq_models[current].num_subsets = nsub
            acq_models[current].subset_num = k 

            # compute masks 
            if ind==0:
                mask = acq_models[current].direct(im_one)
                masks.append(mask)

            # rescale by number of gates
            if num_ms > 1:
                acq_models[current] = ScaledOperator(acq_models[current], 1./num_ms)

    return acq_models, masks

def get_asm_attn(sino, attn, acq_model):
    """Get attn ASM from sino, attn image and acq model."""
    asm_attn = pet.AcquisitionSensitivityModel(attn, acq_model)
    # temporary fix pending attenuation offset fix in STIR:
    # converting attenuation into 'bin efficiency'
    asm_attn.set_up(sino)
    bin_eff = pet.AcquisitionData(sino)
    bin_eff.fill(1.0)
    asm_attn.unnormalise(bin_eff)
    asm_attn = pet.AcquisitionSensitivityModel(bin_eff)
    return asm_attn

if __name__ == "__main__":
    main()