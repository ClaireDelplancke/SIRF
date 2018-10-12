/*
CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
Copyright 2015 - 2017 Rutherford Appleton Laboratory STFC

This is software developed for the Collaborative Computational
Project in Positron Emission Tomography and Magnetic Resonance imaging
(http://www.ccppetmr.ac.uk/).

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

/*!
\file
\ingroup Registration
\brief Base class for all SIRF registration.

The parameter file should have interfile-like syntax.
The variables will be stored as a vector of floats and converted into the required type (int, unsigned int, etc) if necessary.
Multiple variables for a given parameter should be comma separated.
Spaces and tabs will be ignored.
For the title, it doesn't matter what is written as it will be ignored, but something has to be there (otherwise the first parameter will be ignored).
Possible parameters are all the Set<something> methods for each class (e.g., nifty_aladin::SetPerformRigid) and should be written in the parameter file without the "Set" (e.g., PerformRigid).

An example is given below:
    SomeTitle :=
        ReferenceTimePoint := 1
        FloatingTimePoint := 2
        LinearEnergyWeights := 1.5,1
        AdditiveMC :=
    end :=

More examples can be found in data/examples/Registration/paramFiles

\author Richard Brown
\author CCP PETMR
*/

#ifndef _SIRFREG_H_
#define _SIRFREG_H_

#include <stdexcept>
#include <nifti1_io.h>
#include <string>
#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>
#include "NiftiImage3D.h"
#include "NiftiImage3DDeformation.h"
#include "NiftiImage3DDisplacement.h"

namespace sirf {
/// Base class for registration algorithms wrapped by SIRFReg
class SIRFReg
{
public:

    /// Constructor
    SIRFReg() {}

    /// Destructor
    virtual ~SIRFReg() {}

    /// Set parameter file
    void set_parameter_file(const std::string &parameter_filename) { _parameter_filename = parameter_filename; }

    /// Set reference image
    void set_reference_image(const NiftiImage3D &reference_image) { _reference_image = reference_image; }

    /// Set floating image
    void set_floating_image(const NiftiImage3D &floating_image) { _floating_image = floating_image; }

    /// Process
    virtual void process() = 0;

    /// Get registered image
    const NiftiImage3D &get_output() const { return _warped_image; }

    /// Get forward deformation field image
    const NiftiImage3DDeformation  &get_deformation_field_forward()  const { return _def_image_forward; }
    /// Get inverse deformation field image
    const NiftiImage3DDeformation  &get_deformation_field_inverse()  const { return _def_image_inverse; }
    /// Get forward displacement field image
    const NiftiImage3DDisplacement &get_displacement_field_forward() const { return _disp_image_forward; }
    /// Get inverse displacement field image
    const NiftiImage3DDisplacement &get_displacement_field_inverse() const { return _disp_image_inverse; }

    /// Set string parameter. Check if any set methods match the method given by par.
    /// If so, set the value given by arg. Convert to float/int etc., as necessary.
    /// Up to 2 arguments, leave blank if unneeded. These are applied after parsing
    /// the parameter file.
    void set_parameter(const std::string &par, const std::string &arg1 = "", const std::string &arg2 = "");

protected:

    /// Parse parameter file
    virtual void parse_parameter_file() = 0;

    /// Check parameters
    virtual void check_parameters();

    /// Set any extra parameters
    virtual void set_parameters() = 0;

    /// Store extra parameters. Only apply them after parsing.
    std::vector<std::string> _extra_params;

    /// Parameter filename
    boost::filesystem::path _parameter_filename;

    /// Reference image
    NiftiImage3D _reference_image;
    /// Floating image
    NiftiImage3D _floating_image;
    /// Warped image
    NiftiImage3D _warped_image;

    /// Forward displacement field image
    NiftiImage3DDisplacement _disp_image_forward;
    /// Inverse displacement field image
    NiftiImage3DDisplacement _disp_image_inverse;
    /// Forward deformation field image
    NiftiImage3DDeformation _def_image_forward;
    /// Inverse deformation field image
    NiftiImage3DDeformation _def_image_inverse;
};
}

#endif
