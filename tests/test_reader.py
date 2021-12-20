from typing import List, Tuple

import numpy as np
import pydicom
from pydicom._storage_sopclass_uids import SegmentationStorage

from pydicom_seg.reader import SegmentReader, SegmentReadResult


class TestReader:
    def _get_shared_functional_group(
        self,
        pixel_spacing: List[float],
        slice_thickness: float,
        image_orientation_patient: List[int],
    ) -> pydicom.Dataset:

        pixel_measures = pydicom.Dataset()
        pixel_measures.PixelSpacing = pixel_spacing
        pixel_measures.SliceThickness = slice_thickness
        plane_orientation = pydicom.Dataset()
        plane_orientation.ImageOrientationPatient = image_orientation_patient

        shared_functional_group = pydicom.Dataset()
        shared_functional_group.PixelMeasuresSequence = pydicom.Sequence(
            [pixel_measures]
        )
        shared_functional_group.PlaneOrientationSequence = pydicom.Sequence(
            [plane_orientation]
        )
        return shared_functional_group

    def _get_per_frame_functional_group(
        self,
        dimension_index: int,
        image_position_patient: List[float],
        referenced_segment_number: int,
    ) -> pydicom.Dataset:

        frame_content = pydicom.Dataset()
        frame_content.DimensionIndexValues = dimension_index
        plane_position = pydicom.Dataset()
        plane_position.ImagePositionPatient = image_position_patient
        segment_identification = pydicom.Dataset()
        segment_identification.ImagePositionPatient = referenced_segment_number

        per_frame_functional_group = pydicom.Dataset()
        per_frame_functional_group.ReferencedSegmentNumber = pydicom.Sequence(
            [frame_content]
        )
        per_frame_functional_group.PlanePositionSequence = pydicom.Sequence(
            [plane_position]
        )
        per_frame_functional_group.SegmentIdentificationSequence = pydicom.Sequence(
            [segment_identification]
        )
        return per_frame_functional_group

    def _create_read_common_test_dataset(
        self,
    ) -> Tuple[pydicom.Dataset, Tuple[int, int, int]]:
        ds = pydicom.Dataset()
        ds.SOPClassUID = SegmentationStorage
        ds.Modality = "SEG"

        segment = pydicom.Dataset()
        segment.SegmentNumber = 1
        ds.SegmentSequence = pydicom.Sequence([segment])

        slice_thickness = 5
        slice_count = 56
        start_position = -761.7
        rows = 512
        cols = 512
        end_position = np.round(start_position + slice_thickness * (slice_count - 1), 5)
        shape = rows, cols, slice_count

        shared_functional_group = self._get_shared_functional_group(
            pixel_spacing=[1.0, 1.0],
            slice_thickness=slice_thickness,
            image_orientation_patient=[1, 0, -0, -0, 1, 0],
        )
        ds.SharedFunctionalGroupsSequence = pydicom.Sequence([shared_functional_group])

        position_z = np.linspace(start_position, end_position, slice_count)
        positions = np.stack(
            (np.full_like(position_z, 0), np.full_like(position_z, 0), position_z),
            axis=-1,
        ).tolist()

        ds.PerFrameFunctionalGroupsSequence = pydicom.Sequence(
            [
                self._get_per_frame_functional_group(1, position, 1)
                for position in positions
            ]
        )

        ds.Columns = rows
        ds.Rows = cols
        return ds, shape

    def test_read_common(self) -> None:
        ds, shape = self._create_read_common_test_dataset()

        reader = SegmentReader()
        result = SegmentReadResult()
        reader._read_common(ds, result)

        assert np.array_equal(result.size, shape)
