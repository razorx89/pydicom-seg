import logging
from typing import Dict, Tuple

import numpy as np
import pydicom

from pydicom_seg.dicom_utils import dcm_to_sitk_orientation

logger = logging.getLogger(__name__)


def get_segment_map(dataset: pydicom.Dataset) -> Dict[int, pydicom.Dataset]:
    result: Dict[int, pydicom.Dataset] = {}
    last_number = 0
    for segment in dataset.SegmentSequence:
        if segment.SegmentNumber in result:
            raise ValueError(
                f"Segment {segment.SegmentNumber} was declared more than once."
            )

        if segment.SegmentNumber == 0:
            raise ValueError("Segment numbers must be start at 1.")

        if segment.SegmentNumber <= last_number:
            logger.warning(
                "Segment numbering should be monotonically increasing (last=%d, current=%d)",
                last_number,
                segment.SegmentNumber,
            )

        result[segment.SegmentNumber] = segment
        last_number = segment.SegmentNumber
    return result


def get_declared_image_spacing(dataset: pydicom.Dataset) -> Tuple[float, float, float]:
    sfg = dataset.SharedFunctionalGroupsSequence[0]
    if "PixelMeasuresSequence" not in sfg:
        raise ValueError("Pixel measures FG is missing!")

    pixel_measures = sfg.PixelMeasuresSequence[0]
    # DICOM defines (row spacing, column spacing) -> (y, x)
    y_spacing, x_spacing = pixel_measures.PixelSpacing
    if "SpacingBetweenSlices" in pixel_measures:
        z_spacing = pixel_measures.SpacingBetweenSlices
    else:
        z_spacing = pixel_measures.SliceThickness

    return float(x_spacing), float(y_spacing), float(z_spacing)


def get_image_direction(dataset: pydicom.Dataset) -> np.ndarray:
    sfg = dataset.SharedFunctionalGroupsSequence[0]
    if "PlaneOrientationSequence" not in sfg:
        raise ValueError("Plane Orientation (Patient) is missing")

    iop = sfg.PlaneOrientationSequence[0].ImageOrientationPatient
    assert len(iop) == 6

    return dcm_to_sitk_orientation(iop)


def get_image_origin_and_extent(
    dataset: pydicom.Dataset, direction: np.ndarray
) -> Tuple[Tuple[float, ...], float]:
    frames = dataset.PerFrameFunctionalGroupsSequence
    slice_dir = direction[:, 2]
    reference_position = np.asarray(
        [float(x) for x in frames[0].PlanePositionSequence[0].ImagePositionPatient]
    )

    min_distance = 0.0
    origin: Tuple[float, ...] = (0.0, 0.0, 0.0)
    distances: Dict[Tuple, float] = {}
    for frame_idx, frame in enumerate(frames):
        frame_position = tuple(
            float(x) for x in frame.PlanePositionSequence[0].ImagePositionPatient
        )
        if frame_position in distances:
            continue

        frame_distance = np.dot(frame_position - reference_position, slice_dir)
        distances[frame_position] = frame_distance

        if frame_idx == 0 or frame_distance < min_distance:
            min_distance = frame_distance
            origin = frame_position

    # Sort all distances ascending and compute extent from minimum and
    # maximum distance to reference plane
    distance_values = sorted(distances.values())
    extent = 0.0
    if len(distance_values) > 1:
        extent = abs(distance_values[0] - distance_values[-1])

    return origin, extent
