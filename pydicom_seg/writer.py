import datetime
import logging
from typing import Iterable, List

import numpy as np
import pydicom
from pydicom._storage_sopclass_uids import SegmentationStorage
import SimpleITK as sitk

from pydicom_seg import writer_utils


logger = logging.getLogger(__name__)



class MultiClassWriter:
    """Writer for DICOM-SEG files from multi-class segmentations.

    Usage:
    ```python
    segmentation = ...  # A multi-class segmentation as SimpleITK image
    series_dcms = ...  # List of `pydicom.Dataset`s related to the segmentation

    template = pydicom_seg.template.from_dcmqi_metainfo('metainfo.json')
    writer = MultiClassWriter(template)
    dcm = writer.write(segmentation, series_dcms)
    dcm.save_as('<path>')
    ```
    """

    def __init__(self,
                 template: pydicom.Dataset,
                 inplane_cropping: bool = True,
                 skip_empty_slices: bool = True,
                 skip_missing_segment: bool = False):
        """ Constructs a new writer instance.

        Writing DICOM-SEGs can be optimized in respect to the required disk
        space. Empty slices/frames of a 3D volume, containing only zeros, can
        be omitted from the frame sequence. Furthermore, the segmentation might
        only span a small area in a slice and thus can be cropped to the
        minimal enclosing bounding box.

        Args:
            template: A `pydicom.Dataset` holding all relevant meta information
                about the DICOM-SEG series. It has the same meaning as the
                `metainfo.json` file for the dcmqi binaries.
            inplane_cropping: If enabled, slices will be cropped (Rows/Columns)
                to the minimum enclosing boundingbox of all labels across all
                slices.
            skip_empty_slices: If enabled, empty slices with only zeros 
                (background label) will be ommited from the DICOM-SEG.
            skip_missing_segment: If enabled, just emit a warning if segment
                information is missing in the template for a specific label.
                The segment won't be included in the final DICOM-SEG. 
                Otherwise, the encoding is aborted if segment information is
                missing.
        """
        self._inplane_cropping = inplane_cropping
        self._skip_empty_slices = skip_empty_slices
        self._skip_missing_segment = skip_missing_segment
        self._template = template

    def write(self,
              segmentation: sitk.Image,
              source_images: Iterable[pydicom.Dataset]) -> pydicom.Dataset:
        """Writes a DICOM-SEG dataset from a segmentation image and the
        corresponding DICOM source images.

        Args:
            segmentation: A `SimpleITK.Image` with integer labels and a single
                component per spatial location.
            source_images: An iterable of `pydicom.Dataset` which are the
                source images for the segmentation image.
        
        Returns:
            A `pydicom.Dataset` instance with all necessary information and
            meta information for writing the dataset to disk.
        """
        # TODO Add further checks if source images are from the same series
        slice_to_source_images = self._map_source_images_to_segmentation(
            segmentation, source_images
        )

        # Compute unique labels and their respective bounding boxes
        label_statistics_filter = sitk.LabelStatisticsImageFilter()
        label_statistics_filter.Execute(segmentation, segmentation)
        unique_labels = set([x for x in label_statistics_filter.GetLabels() if x != 0])
        if len(unique_labels) == 0:
            raise ValueError('Segmentation does not contain any labels')

        # Check if all present labels where declared in the DICOM template
        declared_segments = set([x.SegmentNumber for x in self._template.SegmentSequence])
        missing_declarations = unique_labels.difference(declared_segments)
        if missing_declarations:
            missing_segment_numbers = ', '.join([str(x) for x in missing_declarations])
            message = (f'Skipping segment(s) {missing_segment_numbers}, since their '
                       'declaration is missing in the DICOM template')
            if not self._skip_missing_segment:
                raise ValueError(message)
            logger.warning(message)
        labels_to_process = unique_labels.intersection(declared_segments)
        if not labels_to_process:
            raise ValueError('No segments found for encoding as DICOM-SEG')

        # Compute bounding boxes for each present label and optionally restrict
        # the volume to serialize to the joined maximum extent
        bboxs = {x: label_statistics_filter.GetBoundingBox(x) for x in labels_to_process}
        if self._inplane_cropping:
            min_x, min_y, _ = np.min([x[::2] for x in bboxs.values()], axis=0).tolist()
            max_x, max_y, _ = (np.max([x[1::2] for x in bboxs.values()], axis=0) + 1).tolist()
            print('Serializing cropped image planes starting at coordinates '\
                  f'({min_x}, {min_y}) with size ({max_x - min_x}, {max_y - min_y})')
        else:
            min_x, min_y = 0, 0
            max_x, max_y = segmentation.GetWidth(), segmentation.GetHeight()
            print(f'Serializing image planes at full size ({max_x}, {max_y})')

        # Create target dataset for storing serialized data
        result = pydicom.Dataset()
        result.file_meta = pydicom.Dataset()
        result.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        result.SOPClassUID = SegmentationStorage
        result.SOPInstanceUID = pydicom.uid.generate_uid()
        result.is_little_endian = True  # TODO Unecessary with pydicom 1.4
        result.is_implicit_VR = False  # TODO Unecessary with pydicom 1.4
        result.fix_meta_info()
        result.Modality = 'SEG'
        writer_utils.set_binary_segmentation(result)
        writer_utils.set_default_dimension_organization(result)
        writer_utils.import_hierarchy(
            target=result,
            reference=source_images[0],
            import_series=False
        )
        writer_utils.copy_segmentation_template(
            target=result,
            template=self._template,
            segments=labels_to_process,
            skip_missing_segment=self._skip_missing_segment
        )
        writer_utils.set_shared_functional_groups_sequence(
            target=result,
            segmentation=segmentation
        )
        result.ReferencedSeriesSequence = []
        result.PerFrameFunctionalGroupsSequence = []

        referenced_source_images = []
        buffer = sitk.GetArrayFromImage(segmentation)
        frames = []
        for segment in labels_to_process:
            print(f'Processing segment {segment}')

            if self._skip_empty_slices:
                bbox = bboxs[segment]
                min_z, max_z = bbox[4], bbox[5] + 1
            else:
                min_z, max_z = 0, segmentation.GetDepth()
            print('Total number of slices that will be processed for segment ' \
                  f'{segment} is {max_z - min_z} (inclusive from {min_z} to {max_z})')

            skipped_slices = []
            for slice_idx in range(min_z, max_z):
                frame_index = (min_x, min_y, slice_idx)
                frame_position = segmentation.TransformIndexToPhysicalPoint(frame_index)
                frame_data = np.equal(buffer[slice_idx, min_y:max_y, min_x:max_x], segment)
                if self._skip_empty_slices and not frame_data.any():
                    skipped_slices.append(slice_idx)
                    continue
                frames.append(frame_data)

                # Build DICOM PerFrameFunctionalGroup item
                frame_fg = pydicom.Dataset()
                if slice_to_source_images[slice_idx]:
                    # Create item for PerFrameFunctionalGroupSequence
                    dis = pydicom.Dataset()
                    dis.SourceImageSequence = writer_utils.create_referenced_source_image_sequence(
                        slice_to_source_images[slice_idx]
                    )
                    dis.DerivationCodeSequence = writer_utils.create_code_sequence(
                        '113076', 'DCM', 'Segmentation'
                    )
                    frame_fg.DerivationImageSequence = [dis]

                    # Add source images to reference list
                    # TODO Improve lookup of already referenced source image
                    for source_image in slice_to_source_images[slice_idx]:
                        if source_image not in referenced_source_images:
                            referenced_source_images.append(source_image)

                frame_fg.FrameContentSequence = [pydicom.Dataset()]
                frame_fg.FrameContentSequence[0].DimensionIndexValues = [
                    segment,  # Segment number
                    slice_idx - min_z + 1  # Slice index within cropped volume
                ]
                frame_fg.PlanePositionSequence = [pydicom.Dataset()]
                frame_fg.PlanePositionSequence[0].ImagePositionPatient = [
                    f'{x:e}' for x in frame_position
                ]
                frame_fg.SegmentIdentificationSequence = [pydicom.Dataset()]
                frame_fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber = segment
                result.PerFrameFunctionalGroupsSequence.append(frame_fg)

            if skipped_slices:
                print(f'Skipped empty slices for segment {segment}: ' \
                      f'{", ".join([str(x) for x in skipped_slices])}')

        # Create ReferencedSeriesSequence
        rss = pydicom.Dataset()
        rss.SeriesInstanceUID = referenced_source_images[0].SeriesInstanceUID
        rss.ReferencedInstanceSequence = []
        for source_image in referenced_source_images:
            dataset = pydicom.Dataset()
            dataset.ReferencedSOPClassUID = source_image.SOPClassUID
            dataset.ReferencedSOPInstanceUID = source_image.SOPInstanceUID
            rss.ReferencedInstanceSequence.append(dataset)
        result.ReferencedSeriesSequence.append(rss)

        # TODO Implement incremental bitpacking writer
        # https://github.com/DCMTK/dcmtk/blob/master/dcmseg/libsrc/segdoc.cc#L1419

        # Encode all frames into a bytearray
        encoded_frames = np.packbits(frames, bitorder='little').tobytes()
        num_encoded_bytes = len(encoded_frames)
        if self._inplane_cropping or self._skip_empty_slices:
            max_encoded_bytes = segmentation.GetWidth() * segmentation.GetHeight() * \
                segmentation.GetDepth() * len(result.SegmentSequence) // 8
            savings = (1 - num_encoded_bytes / max_encoded_bytes) * 100
            print(f'Optimized frame data length is {num_encoded_bytes:,}B ' \
                  f'instead of {max_encoded_bytes:,}B (saved {savings:.2f}%)')

        # TODO Replace with attribute access when pydicom 1.4.0 is released
        result.add_new((0x0062, 0x0013), 'CS', 'NO')  # SegmentsOverlap

        # Set pixel data and information about spatial extents
        result.Rows = max_y - min_y
        result.Columns = max_x - min_x
        result.NumberOfFrames = len(frames)
        result.PixelData = encoded_frames

        # Set timestamps of creation
        timestamp_now = datetime.datetime.now()
        result.SeriesDate = timestamp_now.strftime('%Y%m%d')
        result.SeriesTime = timestamp_now.strftime('%H%M%S.%f')
        result.ContentDate = result.SeriesDate
        result.ContentTime = result.SeriesTime

        return result

    def _map_source_images_to_segmentation(self,
                                           segmentation: sitk.Image,
                                           source_images: Iterable[pydicom.Dataset]
                                           ) -> List[List[pydicom.Dataset]]:
        """Maps an iterable of source image datasets to the slices of a
        SimpleITK image.

        Args:
            segmentation: A `SimpleITK.Image` with integer labels and a single
                component per spatial location.
            source_images: An iterable of `pydicom.Dataset` which are the
                source images for the segmentation image.
        
        Returns:
            A `list` with a `list` for each slice, which contains the mapped
            `pydicom.Dataset` instances for that slice location. Slices can
            have zero or more matched datasets.
        """
        result = [list() for _ in range(segmentation.GetDepth())]
        for source_image in source_images:
            position = [float(x) for x in source_image.ImagePositionPatient]
            index = segmentation.TransformPhysicalPointToIndex(position)
            if index[2] < 0 or index[2] >= segmentation.GetDepth():
                continue
            # TODO Add reverse check if segmentation is contained in image
            result[index[2]].append(source_image)
        slices_mapped = sum(len(x) > 0 for x in result)
        print(f'{slices_mapped} of {segmentation.GetDepth()} slices mapped to source DICOM images')
        return result
