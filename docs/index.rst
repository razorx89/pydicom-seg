Welcome to pydicom-seg's documentation!
=======================================

Converting DICOM-SEG files into ITK compatible data formats, commonly used for
research, is made possible by the `dcmqi <https://github.com/QIICR/dcmqi>`_
project for some time. However, the project is written in C++ and offers only
access to the conversion via the binaries `itkimage2segimage` and
`segimage2itkimage`. After a conversion of a DICOM-SEG file to ITK NRRD file
format, the user has to scan the output directory for generated files, load
them individually and potentially combine multiple files to the desired format.

This library aims to make this process much easier, by providing a Python
native implementation of reading and writing functionality with support for
`numpy` and `SimpleITK`. Additionally, common use cases like loading
multi-class segmentations are supported out-of-the-box.

.. toctree::
    :maxdepth: 2
    :caption: Contents

    install.rst
    guides/index.rst
    api.rst
    license.rst
    citation.rst