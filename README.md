# pydicom-seg

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Decoder and encoder functionality for [DICOM-SEG](http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html) medical image segmentation storage using [pydicom](https://github.com/pydicom/pydicom) as DICOM serialization/deserialization library.

## Installation

This package uses [Poetry](https://python-poetry.org/) as build system.

```bash
git clone https://github.com/razorx89/pydicom-seg
cd pydicom-seg
poetry build
pip3 install dist/pydicom_seg-<version>-py3-none-any.whl
```

## Getting Started

```python
import pydicom
import pydicom_seg
import SimpleITK as sitk

# Load the DICOM-SEG file
dcm = pydicom.dcmread('segmentation.dcm')
# Instantiate a suitable reader for expected segmentation data
reader = pydicom_seg.MultiClassReader()
# Get an ITK image from DICOM (assuming non-overlapping segmentations)
result = reader.read(dcm)
image_data = result.data  # directly available
image = result.image  # lazy construction
# Store segmentation data in a different format or use it for further computations
sitk.WriteImage(image, '/tmp/segmentation.nrrd', True)
```
