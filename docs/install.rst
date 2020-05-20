Installation Guide
==================

``pydicom-seg`` is written for Python 3.5 upwards and does not support older
versions.

Install from PyPI
-----------------

.. code-block:: bash

    pip install pydicom-seg

Install from Github
-------------------

``pydicom-seg`` uses `Poetry <https://python-poetry.org>`_ as build and
dependency management system. Please follow the `installation instructions
<https://python-poetry.org/docs/#installation>`_ on their documentation
website.

First, clone the source code repository somewhere on your filesystem. Make sure
to also clone git submodules, since this package relies on some files from the
`dcmqi <https://github.com/QIICR/dcmqi>`_ project for compatibility
reasons.

.. code-block:: bash

    git clone \
        --recurse-submodules \
        https://github.com/razorx89/pydicom-seg.git
    cd pydicom-seg

.. warning::

    If you do not clone submodules, then the build process will still succeed.
    However, the built package won't contain the required files.

Building and installing the package can be accomplished by:

.. code-block:: bash

    poetry build
    pip install dist/pydicom_seg-0.x.x-py3-none-any.whl
