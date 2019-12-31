import abc
import enum
import logging

import numpy as np
import pydicom
import SimpleITK as sitk

from pydicom_seg import reader_utils


logger = logging.getLogger(__name__)


class AlgorithmType(enum.Enum):
    AUTOMATIC = 'AUTOMATIC'
    SEMIAUTOMATIC = 'SEMIAUTOMATIC'
    MANUAL = 'MANUAL'


class SegmentsOverlap(enum.Enum):
    YES = 'YES'
    UNDEFINED = 'UNDEFINED'
    NO = 'NO'


class SegmentationType(enum.Enum):
    BINARY = 'BINARY'
    FRACTIONAL = 'FRACTIONAL'


class _ReaderBase(abc.ABC):
    def __init__(self, dataset: pydicom.Dataset):
        if dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.66.4' or dataset.Modality != 'SEG':
            raise ValueError('DICOM dataset is not a DICOM-SEG storage')
        self._dataset = dataset

        self._segment_infos = reader_utils.get_segment_map(dataset)
        self._spacing = reader_utils.get_declared_image_spacing(self.dataset)
        self._direction = reader_utils.get_image_direction(self.dataset)
        self._direction.flags.writeable = False
        self._origin, extent = reader_utils.get_image_origin_and_extent(self.dataset, self._direction)
        self._size = (dataset.Columns, dataset.Rows, int(np.ceil(extent / self._spacing[-1]) + 1))

        self._decode()

    @property
    def dataset(self):
        return self._dataset

    @property
    def direction(self):
        return self._direction

    @property
    def origin(self):
        return self._origin

    @property
    def referenced_series_uid(self):
        return self._dataset.ReferencedSeriesSequence[0].SeriesInstanceUID

    @property
    def referenced_instance_uids(self):
        return [
            x.ReferencedSOPInstanceUID
            for x in self._dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence
        ]

    def get_segment_info(self, number):
        return self._segment_infos[number]

    @property
    def size(self):
        return self._size

    @property
    def spacing(self):
        return self._spacing

    @abc.abstractmethod
    def _decode(self):
        pass


class SegmentReader(_ReaderBase):
    def _decode(self):
        # SimpleITK has currently no support for writing slices into memory, allocate a numpy array
        # as intermediate buffer and create an image afterwards
        segmentation_type = SegmentationType[self.dataset.SegmentationType]
        dtype = np.uint8 if segmentation_type == SegmentationType.BINARY else np.float32
        segment_buffer = np.zeros(self.size[::-1], dtype=dtype)

        self._segment_images = {}
        for segment_number in self._segment_infos:
            # Dummy image for computing indices from physical points
            dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
            dummy.SetOrigin(self.origin)
            dummy.SetSpacing(self.spacing)
            dummy.SetDirection(self.direction.ravel())

            # Iterate over all frames and check for referenced segment number
            for frame_idx, pffg in enumerate(self.dataset.PerFrameFunctionalGroupsSequence):
                if segment_number != pffg.SegmentIdentificationSequence[0].ReferencedSegmentNumber:
                    continue
                frame_position = [float(x) for x in pffg.PlanePositionSequence[0].ImagePositionPatient]
                frame_index = dummy.TransformPhysicalPointToIndex(frame_position)
                slice_data = self.dataset.pixel_array[frame_idx]

                # If it is fractional data, then convert to range [0, 1]
                if segmentation_type == SegmentationType.FRACTIONAL:
                    slice_data = slice_data.astype(dtype) / self.dataset.MaximumFractionalValue

                segment_buffer[frame_index[2]] = slice_data

            # Construct final SimpleITK image from numpy array
            image = sitk.GetImageFromArray(segment_buffer)
            image.SetOrigin(self.origin)
            image.SetSpacing(self.spacing)
            image.SetDirection(self.direction.ravel())

            self._segment_images[segment_number] = image

    def get_segment_image(self, number):
        return self._segment_images[number]


class MultiClassReader(_ReaderBase):
    def _decode(self):
        # Multi-class decoding assumes binary segmentations
        segmentation_type = SegmentationType(self.dataset.SegmentationType)
        if segmentation_type != SegmentationType.BINARY:
            raise ValueError('Invalid segmentation type, only BINARY is supported for decoding multi-class segmentations.')

        # Multi-class decoding requires non-overlapping segmentations
        # TODO Replace with attribute access when pydicom 1.4.0 is released
        segments_overlap = self.dataset.get(pydicom.tag.Tag(0x0062, 0x0013))  # SegmentsOverlap
        if segments_overlap is None:
            segments_overlap = SegmentsOverlap.UNDEFINED
            logger.warning('DICOM-SEG does not specify "(0062, 0013) SegmentsOverlap", assuming UNDEFINED and checking pixels')
        else:
            segments_overlap = SegmentsOverlap(segments_overlap.value)

        if segments_overlap == SegmentsOverlap.YES:
            raise ValueError('Segmentation contains overlapping segments, cannot read as multi-class.')

        # Choose suitable data format for multi-class segmentions, depending
        # on the number of segments
        max_segment_number = max(self._segment_infos.keys())
        if max_segment_number < 256:
            dtype = np.uint8
        else:
            dtype = np.uint16

        # SimpleITK has currently no support for writing slices into memory, allocate a numpy array
        # as intermediate buffer and create an image afterwards
        segment_buffer = np.zeros(self.size[::-1], dtype=dtype)

        # Dummy image for computing indices from physical points
        dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
        dummy.SetOrigin(self.origin)
        dummy.SetSpacing(self.spacing)
        dummy.SetDirection(self.direction.ravel())

        # Iterate over all frames and update buffer with segment mask
        for frame_id, pffg in enumerate(self.dataset.PerFrameFunctionalGroupsSequence):
            referenced_segment_number = pffg.SegmentIdentificationSequence[0].ReferencedSegmentNumber
            frame_position = [float(x) for x in pffg.PlanePositionSequence[0].ImagePositionPatient]
            frame_index = dummy.TransformPhysicalPointToIndex(frame_position)
            binary_mask = np.greater(self.dataset.pixel_array[frame_id], 0)
            if segments_overlap == SegmentsOverlap.UNDEFINED and segment_buffer[frame_index[2]][binary_mask].any():
                raise ValueError('Segments are overlapping, cannot decode as multi-class segmentation.')
            segment_buffer[frame_index[2]][binary_mask] = referenced_segment_number

        # Construct final SimpleITK image from numpy array
        self._image = sitk.GetImageFromArray(segment_buffer)
        self._image.SetOrigin(self.origin)
        self._image.SetSpacing(self.spacing)
        self._image.SetDirection(self.direction.ravel())

    @property
    def image(self):
        return self._image
