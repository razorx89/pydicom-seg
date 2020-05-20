Segmentation Types
==================

Multi-class Segmentations
-------------------------

In the case of multi-class segmentations, a pixel or voxel can only belong to a
single class. Thus, all classes depend on each other. This kind of segmentation
is also sometimes called *label map*. One possible application for multi-class
segmentation would be abdominal organ segmentation in CT or MR images. Each
pixel/voxel can only be assigned to a single organ, but with multiple organs
present in the image.

Example
^^^^^^^
.. image:: ../_static/images/seg-binary-multi-class.svg
    :alt: Examplary multi-label segmentation

DICOM Data Elements
^^^^^^^^^^^^^^^^^^^

+-----------------+------------------+--------+
| Tag             | Tag Name         | Value  |
+=================+==================+========+
| (0x0062,0x0001) | SegmentationType | BINARY |
+-----------------+------------------+--------+
| (0x0062,0x0013) | SegmentsOverlap  | NO     |
+-----------------+------------------+--------+
| (0x0028,0x0100) | BitsAllocated    | 1      |
+-----------------+------------------+--------+
| (0x0028,0x0101) | BitsStored       | 1      |
+-----------------+------------------+--------+
| (0x0028,0x0102) | HighBit          | 0      |
+-----------------+------------------+--------+

Multi-label Segmentations
-------------------------

The main difference between multi-class and multi-label segmentation is the
independence of classes. A pixel/voxel can belong to more than a single class.
For instance, one encoded segment in the DICOM-SEG could represent a liver,
while the next segment would represent tumors within the liver. In this case,
both segments could even build a semantic relationship. Maybe an annotator
segments tumors in the abdomen, but does not differentiate between different
organs. Additionally, multiple organs are fully annotated and thus the tumor
location can be derived from the combination of classes.

Example
^^^^^^^
.. image:: ../_static/images/seg-binary-multi-label.svg
    :alt: Examplary multi-label segmentation

DICOM Data Elements
^^^^^^^^^^^^^^^^^^^

+-----------------+------------------+------------------+
| Tag             | Tag Name         | Value            |
+=================+==================+==================+
| (0x0062,0x0001) | SegmentationType | BINARY           |
+-----------------+------------------+------------------+
| (0x0062,0x0013) | SegmentsOverlap  | YES or UNDEFINED |
+-----------------+------------------+------------------+
| (0x0028,0x0100) | BitsAllocated    | 1                |
+-----------------+------------------+------------------+
| (0x0028,0x0101) | BitsStored       | 1                |
+-----------------+------------------+------------------+
| (0x0028,0x0102) | HighBit          | 0                |
+-----------------+------------------+------------------+

Fractional Segmentations
------------------------

The DICOM-SEG standard allows for fractional segmentations, e.g. for 
representing probabilities or the ratio of tissue composition. A floating 
point value between 0.0 and 1.0 will be quantized into an unsigned integer with
8-bits, which results in a minimum step size of ~0.39% between two integer
values. Possible use cases are:
 * Class probabilities of semantic segmentation algorithms
 * Uncertainty estimations of semantic segmentation algorithms
 * Material specific decomposition in Dual-Energy Computed Tomography (DECT)

Example
^^^^^^^

.. image:: ../_static/images/seg-fractional-probability.svg
    :alt: Example for fractional segmentation
    :width: 49%

.. image:: ../_static/images/seg-fractional-uncertainty.svg
    :alt: Example for fractional segmentation
    :width: 49%

DICOM Data Elements
^^^^^^^^^^^^^^^^^^^

+-----------------+----------------------------+--------------------------+
| Tag             | Tag Name                   | Value                    |
+=================+============================+==========================+
| (0x0062,0x0001) | SegmentationType           | FRACTIONAL               |
+-----------------+----------------------------+--------------------------+
| (0x0062,0x0010) | SegmentationFractionalType | PROBABILITY or OCCUPANCY |
+-----------------+----------------------------+--------------------------+
+ (0x0062,0x000e) | MaximumFractionalValue     | 255                      |
+-----------------+----------------------------+--------------------------+
| (0x0028,0x0100) | BitsAllocated              | 8                        |
+-----------------+----------------------------+--------------------------+
| (0x0028,0x0101) | BitsStored                 | 1                        |
+-----------------+----------------------------+--------------------------+
| (0x0028,0x0102) | HighBit                    | 0                        |
+-----------------+----------------------------+--------------------------+