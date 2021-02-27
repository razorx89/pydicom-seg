import pathlib
import tempfile

import numpy as np
import pydicom
import SimpleITK as sitk

from pydicom_seg.dicom_utils import (
    CodeSequence,
    DimensionOrganizationSequence,
    dcm_to_sitk_orientation,
    sitk_to_dcm_orientation,
)


class TestCodeSequence:
    def test_construction(self) -> None:
        seq = CodeSequence("121322", "DCM", "Segmentation")
        assert len(seq) == 1
        assert seq[0].CodeValue == "121322"
        assert seq[0].CodingSchemeDesignator == "DCM"
        assert seq[0].CodeMeaning == "Segmentation"


class TestDimensionOrganizationSequence:
    def test_empty_sequence(self) -> None:
        seq = DimensionOrganizationSequence()
        assert len(seq) == 0

    def test_add_dimension_without_functional_group_pointer(self) -> None:
        seq = DimensionOrganizationSequence()
        seq.add_dimension("ReferencedSegmentNumber")
        assert len(seq) == 1
        assert seq[0].DimensionOrganizationUID == seq[0].DimensionOrganizationUID
        assert seq[0].DimensionIndexPointer == pydicom.tag.Tag(0x0062, 0x000B)
        assert "FunctionalGroupPointer" not in seq[0]

    def test_add_dimension_with_functional_group_pointer(self) -> None:
        seq = DimensionOrganizationSequence()
        seq.add_dimension("ReferencedSegmentNumber", "SegmentIdentificationSequence")
        assert len(seq) == 1
        assert seq[0].DimensionOrganizationUID == seq[0].DimensionOrganizationUID
        assert seq[0].DimensionIndexPointer == pydicom.tag.Tag(0x0062, 0x000B)
        assert seq[0].FunctionalGroupPointer == pydicom.tag.Tag(0x0062, 0x000A)

    def test_add_dimension_with_functional_group_pointer_tag_based(self) -> None:
        seq = DimensionOrganizationSequence()
        seq.add_dimension(
            pydicom.tag.Tag(0x0062, 0x000B), pydicom.tag.Tag(0x0062, 0x000A)
        )
        assert len(seq) == 1
        assert seq[0].DimensionOrganizationUID == seq[0].DimensionOrganizationUID
        assert seq[0].DimensionIndexPointer == pydicom.tag.Tag(0x0062, 0x000B)
        assert seq[0].FunctionalGroupPointer == pydicom.tag.Tag(0x0062, 0x000A)

    def test_add_multiple_dimensions_copies_same_organization_uid(self) -> None:
        seq = DimensionOrganizationSequence()
        seq.add_dimension("ReferencedSegmentNumber", "SegmentIdentificationSequence")
        seq.add_dimension("ImagePositionPatient", "PlanePositionSequence")
        assert len(seq) == 2
        assert seq[0].DimensionOrganizationUID == seq[1].DimensionOrganizationUID


class TestOrientationConversions:
    def setup(self) -> None:
        dcm_path = str(
            pathlib.Path(__file__).parent.parent
            / "pydicom_seg"
            / "externals"
            / "dcmqi"
            / "data"
            / "segmentations"
            / "ct-3slice"
            / "01.dcm"
        )
        self.dcm = pydicom.dcmread(dcm_path)
        self.dcm.ImageOrientationPatient = [
            "1.000000e+00",
            "0.000000e+00",
            "0.000000e+00",
            "0.000000e+00",
            "-1.000000e+00",
            "0.000000e+00",
        ]
        self.tmp_file = tempfile.NamedTemporaryFile(suffix=".dcm")
        self.dcm.save_as(self.tmp_file.name)

    def teardown(self) -> None:
        self.tmp_file.close()

    def test_dcm_to_sitk_conversion(self) -> None:
        img = sitk.ReadImage(self.tmp_file.name)
        converted_orientation = dcm_to_sitk_orientation(
            self.dcm.ImageOrientationPatient
        )
        assert np.isclose(img.GetDirection(), converted_orientation.ravel()).all()

    def test_sitk_to_dcm_conversion(self) -> None:
        img = sitk.ReadImage(self.tmp_file.name)
        converted_orientation = sitk_to_dcm_orientation(img)
        dcm_orientation_str = [
            x.original_string for x in self.dcm.ImageOrientationPatient
        ]
        assert dcm_orientation_str == converted_orientation
