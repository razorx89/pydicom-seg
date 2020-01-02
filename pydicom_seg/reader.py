import abc
import enum
import logging
from typing import Dict

import attr
import numpy as np
import pydicom
from pydicom._storage_sopclass_uids import SegmentationStorage
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


# TODO Improve attrs/type-hint usage. pylint produces a lot of false-positives
@attr.s(init=False)
class _ReadResultBase:
    dataset: pydicom.Dataset = attr.ib()
    direction: np.ndarray = attr.ib()
    origin: tuple = attr.ib()
    segment_infos: Dict[int, pydicom.Dataset] = attr.ib()
    size: tuple = attr.ib()
    spacing: tuple = attr.ib()

    @property
    def referenced_series_uid(self):
        return self.dataset.ReferencedSeriesSequence[0].SeriesInstanceUID

    @property
    def referenced_instance_uids(self):
        return [
            x.ReferencedSOPInstanceUID
            for x in self.dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence
        ]


@attr.s(init=False)
class SegmentReadResult(_ReadResultBase):
    _segment_data: Dict[int, np.ndarray] = attr.ib()

    def segment_data(self, number: int) -> np.ndarray:
        return self._segment_data[number]

    def segment_image(self, number: int) -> sitk.Image:
        result = sitk.GetImageFromArray(self._segment_data[number])
        result.SetOrigin(self.origin)
        result.SetSpacing(self.spacing)
        result.SetDirection(self.direction.ravel())
        return result


class MultiClassReadResult(_ReadResultBase):
    data: np.ndarray = attr.ib()

    @property
    def image(self) -> sitk.Image:
        result = sitk.GetImageFromArray(self.data)
        result.SetOrigin(self.origin)
        result.SetSpacing(self.spacing)
        result.SetDirection(self.direction.ravel())
        return result


class _ReaderBase(abc.ABC):
    @abc.abstractmethod
    def read(self, dataset: pydicom.Dataset):
        pass

    def _read_common(self,
                     dataset: pydicom.Dataset,
                     result: _ReadResultBase):
        if dataset.SOPClassUID != SegmentationStorage or dataset.Modality != 'SEG':
            raise ValueError('DICOM dataset is not a DICOM-SEG storage')

        result.dataset = dataset
        result.segment_infos = reader_utils.get_segment_map(dataset)
        result.spacing = reader_utils.get_declared_image_spacing(dataset)
        result.direction = reader_utils.get_image_direction(dataset)
        result.direction.flags.writeable = False
        result.origin, extent = reader_utils.get_image_origin_and_extent(dataset, result.direction)
        result.size = (dataset.Columns, dataset.Rows, int(np.ceil(extent / result.spacing[-1]) + 1))

        return result


class SegmentReader(_ReaderBase):
    def read(self, dataset: pydicom.Dataset) -> SegmentReadResult:
        result = SegmentReadResult()
        self._read_common(dataset, result)

        # SimpleITK has currently no support for writing slices into memory, allocate a numpy array
        # as intermediate buffer and create an image afterwards
        segmentation_type = SegmentationType[dataset.SegmentationType]
        dtype = np.uint8 if segmentation_type == SegmentationType.BINARY else np.float32
        segment_buffer = np.zeros(result.size[::-1], dtype=dtype)

        # pydicom decodes single-frame pixel data without a frame dimension
        frame_pixel_array = dataset.pixel_array
        if dataset.NumberOfFrames == 1 and len(frame_pixel_array.shape) == 2:
            frame_pixel_array = np.expand_dims(frame_pixel_array, axis=0)

        result._segment_data = {}
        for segment_number in result.segment_infos:
            # Dummy image for computing indices from physical points
            dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
            dummy.SetOrigin(result.origin)
            dummy.SetSpacing(result.spacing)
            dummy.SetDirection(result.direction.ravel())

            # Iterate over all frames and check for referenced segment number
            for frame_idx, pffg in enumerate(dataset.PerFrameFunctionalGroupsSequence):
                if segment_number != pffg.SegmentIdentificationSequence[0].ReferencedSegmentNumber:
                    continue
                frame_position = [float(x) for x in pffg.PlanePositionSequence[0].ImagePositionPatient]
                frame_index = dummy.TransformPhysicalPointToIndex(frame_position)
                slice_data = frame_pixel_array[frame_idx]

                # If it is fractional data, then convert to range [0, 1]
                if segmentation_type == SegmentationType.FRACTIONAL:
                    slice_data = slice_data.astype(dtype) / dataset.MaximumFractionalValue

                segment_buffer[frame_index[2]] = slice_data

            result._segment_data[segment_number] = segment_buffer.copy()

        return result


class MultiClassReader(_ReaderBase):
    def read(self, dataset: pydicom.Dataset) -> MultiClassReadResult:
        result = MultiClassReadResult()
        self._read_common(dataset, result)

        # Multi-class decoding assumes binary segmentations
        segmentation_type = SegmentationType(dataset.SegmentationType)
        if segmentation_type != SegmentationType.BINARY:
            raise ValueError('Invalid segmentation type, only BINARY is supported for decoding multi-class segmentations.')

        # Multi-class decoding requires non-overlapping segmentations
        # TODO Replace with attribute access when pydicom 1.4.0 is released
        segments_overlap = dataset.get(pydicom.tag.Tag(0x0062, 0x0013))  # SegmentsOverlap
        if segments_overlap is None:
            segments_overlap = SegmentsOverlap.UNDEFINED
            logger.warning('DICOM-SEG does not specify "(0062, 0013) SegmentsOverlap", assuming UNDEFINED and checking pixels')
        else:
            segments_overlap = SegmentsOverlap(segments_overlap.value)

        if segments_overlap == SegmentsOverlap.YES:
            raise ValueError('Segmentation contains overlapping segments, cannot read as multi-class.')

        # Choose suitable data format for multi-class segmentions, depending
        # on the number of segments
        max_segment_number = max(result.segment_infos.keys())
        if max_segment_number < 256:
            dtype = np.uint8
        else:
            dtype = np.uint16

        # SimpleITK has currently no support for writing slices into memory, allocate a numpy array
        # as intermediate buffer and create an image afterwards
        segment_buffer = np.zeros(result.size[::-1], dtype=dtype)

        # Dummy image for computing indices from physical points
        dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
        dummy.SetOrigin(result.origin)
        dummy.SetSpacing(result.spacing)
        dummy.SetDirection(result.direction.ravel())

        # pydicom decodes single-frame pixel data without a frame dimension
        frame_pixel_array = dataset.pixel_array
        if dataset.NumberOfFrames == 1 and len(frame_pixel_array.shape) == 2:
            frame_pixel_array = np.expand_dims(frame_pixel_array, axis=0)

        # Iterate over all frames and update buffer with segment mask
        for frame_id, pffg in enumerate(dataset.PerFrameFunctionalGroupsSequence):
            referenced_segment_number = pffg.SegmentIdentificationSequence[0].ReferencedSegmentNumber
            frame_position = [float(x) for x in pffg.PlanePositionSequence[0].ImagePositionPatient]
            frame_index = dummy.TransformPhysicalPointToIndex(frame_position)
            binary_mask = np.greater(frame_pixel_array[frame_id], 0)
            if segments_overlap == SegmentsOverlap.UNDEFINED and segment_buffer[frame_index[2]][binary_mask].any():
                raise ValueError('Segments are overlapping, cannot decode as multi-class segmentation.')
            segment_buffer[frame_index[2]][binary_mask] = referenced_segment_number

        # Construct final SimpleITK image from numpy array
        result.data = segment_buffer

        return result
