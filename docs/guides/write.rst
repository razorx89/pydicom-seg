Writing DICOM-SEG
-----------------

.. code-block:: python

    import pydicom
    import pydicom_seg
    import SimpleITK as sitk

Templates
^^^^^^^^^

For writing DICOM-SEG files, some information besides the segmentation data is
needed in order to fully comply to the DICOM standard. On the one hand, some
information about the content creator and series level DICOM data elements need
to be specified. On the other hand, each segment to encode has to be specified
including codes for automatical identification of the segmented tissue.

Since this is normally done in advance, before creating DICOM-SEGs
automatically, it is recommended to use the `web-based editor <http://qiicr.org/dcmqi/#/seg>`_
from the `dcmqi <https://github.com/QIICR/dcmqi>`_ project. The resulting
``metainfo.json`` file can be loaded using a compatibility importer function:

.. code-block:: python

    template = pydicom_seg.template.from_dcmqi_metainfo('metainfo.json')


Multi-class segmentations
^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-class segmentations can be written using the ``MultiClassWriter``, which
has a few customization options available. These options can drastically reduce
the generated file size if the segmentation data is very sparse.

.. code-block:: python

    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=True,  # Crop image slices to the minimum bounding box on 
                                # x and y axes
        skip_empty_slices=True,  # Don't encode slices with only zeros
        skip_missing_segment=False,  # If a segment definition is missing in the
                                     # template, then raise an error instead of
                                     # skipping it.
    )

Inplane cropping is performed over all segments present in the segmentation
data. Since the DICOM standard requires encoded frames to have all the same
size, the minimum bounding box of all segments is calculated for the x and
y axes.

.. warning::

    Inplane cropping is an experimental feature and might not be supported by other
    frameworks or DICOM viewers.

If enabled, slices which have no annotation present for a given segment can
be skipped during encoding. This results in smaller file sizes, because
otherwise a frame would be encoded with only zero values.

Exemplary Workflow
^^^^^^^^^^^^^^^^^^

In order to illustrate the creation of a DICOM-SEG from segmentation data, a
typical image analysis workflow is shown in the following. First, an image is
loaded from disk as a ``SimpleITK.Image``.

.. code-block:: python

    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames('dir/to/study', '<SeriesInstanceUID>')
    reader.SetFileNames(dcm_files)
    image = reader.Execute()
    image_data = sitk.GetArrayFromImage(image)

At this point some kind of image processing is taking place, maybe a semantic
segmentation neural network is applied in order to get the segmentation data.

.. code-block:: python
    
    segmentation_data = process_image(image_data)

Since ``pydicom_seg`` expects a ``SimpleITK.Image`` as input to the writer, the
segmentation data needs to be encapsulated with all relevant information about
the image grid. This is very important for encoding the segments and
referencing the corresponding source DICOM files.

.. code-block:: python

    segmentation = sitk.GetImageFromArray(segmentation_data)
    segmentation.CopyInformation(image)

Finally, the actual creation of the DICOM-SEG can be performed. Therefore the
source DICOM files need to be loaded with ``pydicom``, but for optimization
purposes the pixel data can be skipped.

.. code-block:: python

    source_images = [
        pydicom.dcmread(x, stop_before_pixels=True)
        for x in dcm_files
    ]
    dcm = writer.write(segmentation, source_images)

The created DICOM dataset can now also be modified, e.g. setting some custom
UIDs instead of random generated UIDs, or private tags with additional
data. Lastly, the dataset can be stored on disk:

.. code-block:: python

    dcm.save_as('segmentation.dcm')

Or the dataset can be directly stored in a research PACS (e.g.
`Orthanc <https://www.orthanc-server.com/>`_) using DICOM-Web technology and
the Python package `dicomweb-client <https://github.com/MGHComputationalPathology/dicomweb-client>`_:

.. code-block:: python

    from dicomweb_client.api import DICOMwebClient
    client = DICOMwebClient('https://<some-pacs-server>')
    client.store_instances(datasets=[dcm])
