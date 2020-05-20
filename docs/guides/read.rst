Reading DICOM-SEG
-----------------

.. code-block:: python

    import pydicom
    import pydicom_seg
    import SimpleITK as sitk

All readers are stateless and return a result object holding information and
data from a DICOM-SEG file. The segmentation data is directly available as a
numpy array. However, if you need additional processing, e.g. resampling to a
different image grid, then the result objects provide utility functionality to
construct a corresponding ``SimpleITK.Image``.

Multi-class segmentations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    dcm = pydicom.dcmread('multi-class-seg.dcm')

    reader = pydicom_seg.MultiClassReader()
    result = reader.read(dcm)

    image_data = result.data  # directly available
    image = result.image  # lazy construction
    sitk.WriteImage(image, '/tmp/segmentation.nrrd', True)

Binary segments
^^^^^^^^^^^^^^^

In DICOM-SEG segments are always encoded independently for multi-label
segmentations. Instead of a single numpy array or SimpleITK image, multiple
segments with their associated segment number are decoded and stored in the
result object.

.. code-block:: python

    dcm = pydicom.dcmread('multi-label-seg.dcm')

    reader = pydicom_seg.SegmentReader()
    result = reader.read(dcm)

    for segment_number in result.available_segments:
        image_data = result.segment_data(segment_number)  # directly available
        image = result.segment_image(segment_number)  # lazy construction
        sitk.WriteImage(image, f'/tmp/seg-{segment_number}.nrrd', True)

Fractional segments
^^^^^^^^^^^^^^^^^^^

Fractional segmentations use the same reader as multi-label segmentations,
however the decoded data is of a floating point instead of an integer data
type.

.. code-block:: python

    dcm = pydicom.dcmread('fractional-seg.dcm')

    reader = pydicom_seg.SegmentReader()
    result = reader.read(dcm)

    for segment_number in result.available_segments:
        image_data = result.segment_data(segment_number)  # directly available
        assert isinstance(image_data.dtype, float)
