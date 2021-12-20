import abc
import logging
from typing import Dict, List, Set

import attr
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom._storage_sopclass_uids import SegmentationStorage

from pydicom_seg import reader_utils
from pydicom_seg.segmentation_dataset import SegmentationType, SegmentsOverlap

logger = logging.getLogger(__name__)


# TODO Improve attrs/type-hint usage. pylint produces a lot of false-positives
@attr.s(init=False)
class _ReadResultBase:
    """Base data class for read results.

    Contains common information about a decoded segmentation, e.g. origin,
    voxel spacing and the direction matrix.
    """

    dataset: pydicom.Dataset = attr.ib()
    direction: np.ndarray = attr.ib()
    origin: tuple = attr.ib()
    segment_infos: Dict[int, pydicom.Dataset] = attr.ib()
    size: tuple = attr.ib()
    spacing: tuple = attr.ib()

    @property
    def referenced_series_uid(self) -> str:
        uid: str = self.dataset.ReferencedSeriesSequence[0].SeriesInstanceUID
        return uid

    @property
    def referenced_instance_uids(self) -> List[str]:
        return [
            x.ReferencedSOPInstanceUID
            for x in self.dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence
        ]


@attr.s(init=False)
class SegmentReadResult(_ReadResultBase):
    """Read result for segment-based decoding of DICOM-SEGs."""

    _segment_data: Dict[int, np.ndarray] = attr.ib()

    @property
    def available_segments(self) -> Set[int]:
        return set(self._segment_data.keys())

    def segment_data(self, number: int) -> np.ndarray:
        return self._segment_data[number]

    def segment_image(self, number: int) -> sitk.Image:
        result = sitk.GetImageFromArray(self._segment_data[number])
        result.SetOrigin(self.origin)
        result.SetSpacing(self.spacing)
        result.SetDirection(self.direction.ravel())
        return result


class MultiClassReadResult(_ReadResultBase):
    """Read result for multi-class decoding of DICOM-SEGs."""

    data: np.ndarray = attr.ib()

    @property
    def image(self) -> sitk.Image:
        result = sitk.GetImageFromArray(self.data)
        result.SetOrigin(self.origin)
        result.SetSpacing(self.spacing)
        result.SetDirection(self.direction.ravel())
        return result


class _ReaderBase(abc.ABC):
    """Base class for reader implementations.

    Reading DICOM-SEGs as different output formats still shares a lot of common
    information decoding. This baseclass extracts this common knowledge and
    sets the respective attributes in a `_ReadResultBase` derived result
    instance.
    """

    @abc.abstractmethod
    def read(self, dataset: pydicom.Dataset) -> _ReadResultBase:
        """Read from a DICOM-SEG file.

        Args:
            dataset: A `pydicom.Dataset` with DICOM-SEG content.

        Returns:
            Result object with decoded numpy data and common information about
            the spatial location and extent of the volume.
        """

    def _read_common(self, dataset: pydicom.Dataset, result: _ReadResultBase) -> None:
        """Read common information from a dataset and store it.

        Args:
            dataset: A `pydicom.Dataset` with DICOM-SEG content.
            result: A `_ReadResultBase` derived result object, where the common
                informations are stored.
        """
        if dataset.SOPClassUID != SegmentationStorage or dataset.Modality != "SEG":
            raise ValueError("DICOM dataset is not a DICOM-SEG storage")

        result.dataset = dataset
        result.segment_infos = reader_utils.get_segment_map(dataset)
        result.spacing = reader_utils.get_declared_image_spacing(dataset)
        result.direction = reader_utils.get_image_direction(dataset)
        result.direction.flags.writeable = False
        result.origin, extent = reader_utils.get_image_origin_and_extent(
            dataset, result.direction
        )
        result.size = (
            dataset.Columns,
            dataset.Rows,
            int(np.rint(extent / result.spacing[-1])) + 1,
        )


class SegmentReader(_ReaderBase):
    """Reads binary segments from a DICOM-SEG file.

    All segments in a DICOM-SEG file cover the same spatial extent, but might
    overlap. If a user wants to use each segment individually as a binary
    segmentation, then this reader extracts all segments as individual numpy
    arrays. The read operation creates a `SegmentReadResult` object with common
    information about the spatial location and extent shared by all segments,
    as well as the binary segmentation data for each segment.

    Example:
        ::

            dcm = pydicom.dcmread('segmentation.dcm')
            reader = pydicom_seg.SegmentReader()
            result = reader.read(dcm)
            data = result.segment_data(1)  # numpy array
            image = result.segment_image(1)  # SimpleITK image
    """

    def read(self, dataset: pydicom.Dataset) -> SegmentReadResult:
        result = SegmentReadResult()
        self._read_common(dataset, result)

        # SimpleITK has currently no support for writing slices into memory, allocate a numpy array
        # as intermediate buffer and create an image afterwards
        segmentation_type = SegmentationType[dataset.SegmentationType]
        dtype = np.uint8 if segmentation_type == SegmentationType.BINARY else np.float32

        # pydicom decodes single-frame pixel data without a frame dimension
        frame_pixel_array = dataset.pixel_array
        if dataset.NumberOfFrames == 1 and len(frame_pixel_array.shape) == 2:
            frame_pixel_array = np.expand_dims(frame_pixel_array, axis=0)

        result._segment_data = {}
        for segment_number in result.segment_infos:
            # Segment buffer should be cleared for each segment since
            # segments may have different number of frames!
            segment_buffer = np.zeros(result.size[::-1], dtype=dtype)

            # Dummy image for computing indices from physical points
            dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
            dummy.SetOrigin(result.origin)
            dummy.SetSpacing(result.spacing)
            dummy.SetDirection(result.direction.ravel())

            # get segment ID sequence for the case it is the same for all frames (e.g. only one segment)
            shared_sis = dataset.SharedFunctionalGroupsSequence[0].get(
                "SegmentIdentificationSequence"
            )

            # Iterate over all frames and check for referenced segment number
            for frame_idx, pffg in enumerate(dataset.PerFrameFunctionalGroupsSequence):
                sis = pffg.get(
                    "SegmentIdentificationSequence", shared_sis
                )  # shared_sis as default value
                if segment_number != sis[0].ReferencedSegmentNumber:
                    continue

                frame_position = [
                    float(x) for x in pffg.PlanePositionSequence[0].ImagePositionPatient
                ]
                frame_index = dummy.TransformPhysicalPointToIndex(frame_position)
                slice_data = frame_pixel_array[frame_idx]

                # If it is fractional data, then convert to range [0, 1]
                if segmentation_type == SegmentationType.FRACTIONAL:
                    slice_data = (
                        slice_data.astype(dtype) / dataset.MaximumFractionalValue
                    )

                segment_buffer[frame_index[2]] = slice_data

            result._segment_data[segment_number] = segment_buffer

        return result


class MultiClassReader(_ReaderBase):
    """Reads multi-class segmentations from a DICOM-SEG file.

    If all segments in a DICOM-SEG don't overlap, then it is save to reduce `n`
    binary segmentations to a single segmentation with an integer index for the
    referenced segment at the spatial location. This is a common use-case,
    especially in computer vision applications, for analysing different regions
    in medical imaging. The read operation creates a `MultiClassReadResult`
    object with information about the spatial location and extent, as well as
    the multi-class segmentation data.

    Example:
        ::

            dcm = pydicom.dcmread('segmentation.dcm')
            reader = pydicom_seg.MultiClassReader()
            result = reader.read(dcm)
            data = result.data  # numpy array
            image = result.image  # SimpleITK image
    """

    def read(self, dataset: pydicom.Dataset) -> MultiClassReadResult:
        result = MultiClassReadResult()
        self._read_common(dataset, result)

        # Multi-class decoding assumes binary segmentations
        segmentation_type = SegmentationType(dataset.SegmentationType)
        if segmentation_type != SegmentationType.BINARY:
            raise ValueError(
                "Invalid segmentation type, only BINARY is supported for decoding multi-class segmentations."
            )

        # Multi-class decoding requires non-overlapping segmentations
        segments_overlap = dataset.get("SegmentsOverlap")
        if segments_overlap is None:
            segments_overlap = SegmentsOverlap.UNDEFINED
            logger.warning(
                'DICOM-SEG does not specify "(0062, 0013) SegmentsOverlap", assuming UNDEFINED and checking pixels'
            )
        else:
            segments_overlap = SegmentsOverlap(segments_overlap)

        if segments_overlap == SegmentsOverlap.YES:
            raise ValueError(
                "Segmentation contains overlapping segments, cannot read as multi-class."
            )

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

        # get segment ID sequence for the case it is the same for all frames (e.g. only one segment)
        shared_sis = dataset.SharedFunctionalGroupsSequence[0].get(
            "SegmentIdentificationSequence"
        )

        # Iterate over all frames and update buffer with segment mask
        for frame_id, pffg in enumerate(dataset.PerFrameFunctionalGroupsSequence):
            sis = pffg.get(
                "SegmentIdentificationSequence", shared_sis
            )  # shared_sis as default value
            referenced_segment_number = sis[0].ReferencedSegmentNumber
            frame_position = [
                float(x) for x in pffg.PlanePositionSequence[0].ImagePositionPatient
            ]
            frame_index = dummy.TransformPhysicalPointToIndex(frame_position)
            binary_mask = np.greater(frame_pixel_array[frame_id], 0)
            if (
                segments_overlap == SegmentsOverlap.UNDEFINED
                and segment_buffer[frame_index[2]][binary_mask].any()
            ):
                raise ValueError(
                    "Segments are overlapping, cannot decode as multi-class segmentation."
                )
            segment_buffer[frame_index[2]][binary_mask] = referenced_segment_number

        # Construct final SimpleITK image from numpy array
        result.data = segment_buffer

        return result
