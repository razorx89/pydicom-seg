import pydicom

from pydicom_seg.reader_utils import get_declared_image_spacing


class TestReaderUtils:
    def _create_empty_pixel_measures_sequence(self) -> pydicom.Dataset:
        ds = pydicom.Dataset()
        ds.SharedFunctionalGroupsSequence = pydicom.Sequence([pydicom.Dataset()])
        ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence = pydicom.Sequence(
            [pydicom.Dataset()]
        )
        return ds

    def test_declared_image_spacing(self) -> None:
        ds = self._create_empty_pixel_measures_sequence()
        pms = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        pms.PixelSpacing = [0.5, 0.5]
        pms.SliceThickness = 1.0

        sx, sy, sz = get_declared_image_spacing(ds)
        assert sx == 0.5
        assert sy == 0.5
        assert sz == 1.0

    def test_declared_image_spacing_unequal_xy(self) -> None:
        ds = self._create_empty_pixel_measures_sequence()
        pms = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        pms.PixelSpacing = [0.5, 0.75]
        pms.SliceThickness = 1.0

        sx, sy, sz = get_declared_image_spacing(ds)
        assert sx == 0.75
        assert sy == 0.5
        assert sz == 1.0
