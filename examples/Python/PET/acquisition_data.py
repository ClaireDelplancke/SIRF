'''Acquisition data handling demo.

Usage:
  acquisition_data [--help | options]

Options:
  -f <file>, --file=<file>     raw data file [default: my_forward_projection.hs]
  -p <path>, --path=<path>     path to data files, defaults to data/examples/PET
                               subfolder of SIRF root folder
  -e <engn>, --engine=<engn>   reconstruction engine [default: STIR]
  -s <stsc>, --storage=<stsc>  acquisition data storage scheme [default: file]
'''

## CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
## Copyright 2015 - 2017 Rutherford Appleton Laboratory STFC
## Copyright 2015 - 2017 University College London.
##
## This is software developed for the Collaborative Computational
## Project in Positron Emission Tomography and Magnetic Resonance imaging
## (http://www.ccppetmr.ac.uk/).
##
## Licensed under the Apache License, Version 2.0 (the "License");
##   you may not use this file except in compliance with the License.
##   You may obtain a copy of the License at
##       http://www.apache.org/licenses/LICENSE-2.0
##   Unless required by applicable law or agreed to in writing, software
##   distributed under the License is distributed on an "AS IS" BASIS,
##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##   See the License for the specific language governing permissions and
##   limitations under the License.

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

import math

from pUtilities import show_2D_array

# import engine module
exec('from sirf.' + args['--engine'] + ' import *')

# process command-line options
data_file = args['--file']
data_path = args['--path']
if data_path is None:
    data_path = examples_data_path('PET')
storage = args['--storage']

# select acquisition data storage scheme
# storage = 'file' (default):
#     all acquisition data generated by the script is kept in
#     scratch files deleted after the script terminates
# storage = 'memory':
#     all acquisition data generated by the script is kept in RAM
#     (avoid if data is very large)
scheme = AcquisitionData.get_storage_scheme()
if scheme != storage:
    print('default storage scheme is %s' % repr(scheme))
    print('setting storage scheme to %s' % repr(storage))
    AcquisitionData.set_storage_scheme(storage)
else:
    print('using default storage scheme %s' % repr(scheme))

def main():

    # direct all engine's messages to files
    msg_red = MessageRedirector('info.txt', 'warn.txt', 'errr.txt')

    # PET acquisition data to be read from this file
    raw_data_file = existing_filepath(data_path, data_file)
    print('raw data: %s' % raw_data_file)
    acq_data = AcquisitionData(raw_data_file)

    # copy the acquisition data into a Python array and display
    dim = acq_data.dimensions()
    print('data dimensions: %d x %d x %d' % dim)
    acq_data.show(range(dim[0]//4))
    acq_array = acq_data.as_array()
    # print('data dimensions: %d x %d x %d' % acq_array.shape)
    # acq_dim = acq_array.shape
    # z = acq_dim[0]//2
    # show_2D_array('Acquisition data', acq_array[z,:,:])

    # rebin the acquisition data
    new_acq_data = acq_data.rebin(3)
    rdim = new_acq_data.dimensions()
    print('rebinned data dimensions: %d x %d x %d' % rdim)
    new_acq_data.show(range(rdim[0]//3), title = 'Rebinned acquisition data')
    #acq_array = new_acq_data.as_array()
    #print('rebinned data dimensions: %d x %d x %d' % acq_array.shape)

    # clone the acquisition data
    new_acq_data = acq_data.clone()
    # display the cloned data
    new_acq_data.show(range(dim[0]//4), title = 'Cloned acquisition data')
    # acq_array = new_acq_data.as_array()
    # show_2D_array('Cloned acquisition data', acq_array[z,:,:])

    print('Checking acquisition data algebra:')
    s = acq_data.norm()
    t = acq_data.dot(acq_data)
##    t = acq_data * acq_data
    print('norm of acq_data.as_array(): %f' % numpy.linalg.norm(acq_array))
    print('acq_data.norm(): %f' % s)
    print('sqrt(acq_data.dot(acq_data)): %f' % math.sqrt(t))
    diff = new_acq_data - acq_data
    print('norm of acq_data.clone() - acq_data: %f' % diff.norm())
    acq_factor = acq_data.get_uniform_copy(0.1)
    new_acq_data = acq_data / acq_factor
##    new_acq_data = acq_data * 10.0
    print('norm of acq_data*10: %f' % new_acq_data.norm())

    # display the scaled data
    new_acq_data.show(range(dim[0]//4), title = 'Scaled acquisition data')
    # acq_array = new_acq_data.as_array()
    # show_2D_array('Scaled acquisition data', acq_array[z,:,:])

    print('Checking images algebra:')
    image = acq_data.create_uniform_image(10.0)
    image_array = image.as_array()
    print('image dimensions: %d x %d x %d' % image_array.shape)
    s = image.norm()
    t = image.dot(image)
##    t = image * image
    print('norm of image.as_array(): %f' % numpy.linalg.norm(image_array))
    print('image.norm(): %f' % s)
    print('sqrt(image.dot(image)): %f' % math.sqrt(t))
    image_factor = image.get_uniform_copy(0.1)
    image = image / image_factor
##    image = image*10
    print('norm of image*10: %f' % image.norm())
    diff = image.clone() - image
    print('norm of image.clone() - image: %f' % diff.norm())

    print('image voxel sizes:')
    print(image.voxel_sizes())
    print('image transform matrix:')
    tmx = image.transf_matrix()
    print(tmx)

try:
    main()
    print('done')
except error as err:
    print('%s' % err.value)

if scheme != storage:
    AcquisitionData.set_storage_scheme(scheme)
