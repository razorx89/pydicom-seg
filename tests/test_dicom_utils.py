import pydicom

from pydicom_seg.dicom_utils import CodeSequence, DimensionOrganizationSequence


class TestCodeSequence:
    def test_construction(self):
        seq = CodeSequence('121322', 'DCM', 'Segmentation')
        assert len(seq) == 1
        assert seq[0].CodeValue == '121322'
        assert seq[0].CodingSchemeDesignator == 'DCM'
        assert seq[0].CodeMeaning == 'Segmentation'


class TestDimensionOrganizationSequence:
    def test_empty_sequence(self):
        seq = DimensionOrganizationSequence()
        assert len(seq) == 0

    def test_add_dimension_without_functional_group_pointer(self):
        seq = DimensionOrganizationSequence()
        seq.add_dimension('ReferencedSegmentNumber')
        assert len(seq) == 1
        assert seq[0].DimensionOrganizationUID == seq[0].DimensionOrganizationUID
        assert seq[0].DimensionIndexPointer == pydicom.tag.Tag(0x0062, 0x000b)
        assert 'FunctionalGroupPointer' not in seq[0]

    def test_add_dimension_with_functional_group_pointer(self):
        seq = DimensionOrganizationSequence()
        seq.add_dimension('ReferencedSegmentNumber', 'SegmentIdentificationSequence')
        assert len(seq) == 1
        assert seq[0].DimensionOrganizationUID == seq[0].DimensionOrganizationUID
        assert seq[0].DimensionIndexPointer == pydicom.tag.Tag(0x0062, 0x000b)
        assert seq[0].FunctionalGroupPointer == pydicom.tag.Tag(0x0062, 0x000a)

    def test_add_dimension_with_functional_group_pointer_tag_based(self):
        seq = DimensionOrganizationSequence()
        seq.add_dimension(pydicom.tag.Tag(0x0062, 0x000b), pydicom.tag.Tag(0x0062, 0x000a))
        assert len(seq) == 1
        assert seq[0].DimensionOrganizationUID == seq[0].DimensionOrganizationUID
        assert seq[0].DimensionIndexPointer == pydicom.tag.Tag(0x0062, 0x000b)
        assert seq[0].FunctionalGroupPointer == pydicom.tag.Tag(0x0062, 0x000a)

    def test_add_multiple_dimensions_copies_same_organization_uid(self):
        seq = DimensionOrganizationSequence()
        seq.add_dimension('ReferencedSegmentNumber', 'SegmentIdentificationSequence')
        seq.add_dimension('ImagePositionPatient', 'PlanePositionSequence')
        assert len(seq) == 2
        assert seq[0].DimensionOrganizationUID == seq[1].DimensionOrganizationUID
