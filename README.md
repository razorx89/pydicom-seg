# pydicom-seg

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/pypi/pyversions/pydicom-seg.svg)](https://img.shields.io/pypi/pyversions/pydicom-seg.svg)
[![PyPI version](https://badge.fury.io/py/pydicom-seg.svg)](https://badge.fury.io/py/pydicom-seg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3597421.svg)](https://doi.org/10.5281/zenodo.3597421)

Reading and writing of [DICOM-SEG](http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html) medical image segmentation storage files using [pydicom](https://github.com/pydicom/pydicom) as DICOM serialization/deserialization library.

## Motivation

Converting DICOM-SEG files into ITK compatible data formats, commonly used for
research, is made possible by the [dcmqi](https://github.com/QIICR/dcmqi)
project for some time. However, the project is written in C++ and offers only
access to the conversion via the binaries `itkimage2segimage` and
`segimage2itkimage`. After a conversion of a DICOM-SEG file to ITK NRRD file
format, the user has to scan the output directory for generated files, load
them individually and potentially combine multiple files to the desired format.

This library aims to make this process much easier, by providing a Python
native implementation of reading and writing functionality with support for
`numpy` and `SimpleITK`. Additionally, common use cases like loading
multi-class segmentations are supported out-of-the-box.

## Installation

### Install from PyPI

```bash
pip install pydicom-seg
```

### Install from source

This package uses [Poetry](https://python-poetry.org/) (version >= 1.0.5) as build system.

```bash
git clone https://github.com/razorx89/pydicom-seg.git
cd pydicom-seg
poetry build
pip install dist/pydicom_seg-<version>-py3-none-any.whl
```

## Getting Started

### Loading binary segments

```python
import pydicom
import pydicom_seg
import SimpleITK as sitk

dcm = pydicom.dcmread('segmentation.dcm')

reader = pydicom_seg.SegmentReader()
result = reader.read(dcm)

for segment_number in result.available_segments:
    image_data = result.segment_data(segment_number)  # directly available
    image = result.segment_image(segment_number)  # lazy construction
    sitk.WriteImage(image, f'/tmp/segmentation-{segment_number}.nrrd', True)
```

### Loading a multi-class segmentation

```python
dcm = pydicom.dcmread('segmentation.dcm')

reader = pydicom_seg.MultiClassReader()
result = reader.read(dcm)

image_data = result.data  # directly available
image = result.image  # lazy construction
sitk.WriteImage(image, '/tmp/segmentation.nrrd', True)
```

### Saving a multi-class segmentation

```python
segmentation: SimpleITK.Image = ...  # A segmentation image with integer data type
                                     # and a single component per voxel
dicom_series_paths = [...]  # Paths to an imaging series related to the segmentation
source_images = [
    pydicom.dcmread(x, stop_before_pixels=True)
    for x in dicom_series_paths
]
template = pydicom_seg.template.from_dcmqi_metainfo('metainfo.json')
writer = pydicom_seg.MultiClassWriter(
    template=template,
    inplane_cropping=True,  # Crop image slices to the minimum bounding box on 
                            # x and y axes
    skip_empty_slices=True,  # Don't encode slices with only zeros
    skip_missing_segments=False,  # If a segment definition is missing in the
                                  # template, then raise an error instead of
                                  # skipping it.
)
dcm = writer.write(segmentation, source_images)
dcm.save_as('segmentation.dcm')
```

## License

`pydicom-seg` is distributed under the [MIT license](./LICENSE).