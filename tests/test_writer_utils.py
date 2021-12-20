import numpy as np
import pydicom
import SimpleITK as sitk

from pydicom_seg.reader_utils import get_declared_image_spacing
from pydicom_seg.writer_utils import set_shared_functional_groups_sequence


class TestWriterUtils:
    def test_pixel_spacing_ordering(self) -> None:
        ds = pydicom.Dataset()
        seg = sitk.Image(1, 1, 1, sitk.sitkInt16)
        seg.SetSpacing((0.75, 1.5, 5.0))
        set_shared_functional_groups_sequence(ds, seg)

        spacing = (
            ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        )
        assert np.isclose(float(spacing[0]), 1.5)
        assert np.isclose(float(spacing[1]), 0.75)

    def test_pixel_spacing_round_trip(self) -> None:
        ds = pydicom.Dataset()
        seg = sitk.Image(1, 1, 1, sitk.sitkInt16)
        spacing = (0.75, 1.5, 5.0)
        seg.SetSpacing(spacing)
        set_shared_functional_groups_sequence(ds, seg)
        new_spacing = get_declared_image_spacing(ds)

        assert new_spacing == spacing
