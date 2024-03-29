import tempfile

import numpy as np
import pydicom
import pytest

from pydicom_seg import __version__
from pydicom_seg.dicom_utils import DimensionOrganizationSequence
from pydicom_seg.segmentation_dataset import (
    SegmentationDataset,
    SegmentationFractionalType,
    SegmentationType,
)


class TestSegmentationDataset:
    def setup(self) -> None:
        self.dataset = SegmentationDataset(
            rows=1, columns=1, segmentation_type=SegmentationType.BINARY
        )
        self.setup_dummy_segment(self.dataset)

    def setup_dummy_segment(self, dataset: pydicom.Dataset) -> None:
        ds = pydicom.Dataset()
        ds.SegmentNumber = 1
        dataset.SegmentSequence.append(ds)

    def generate_dummy_source_image(self) -> pydicom.Dataset:
        ds = pydicom.Dataset()
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        return ds

    def test_dataset_is_writable(self) -> None:
        with tempfile.NamedTemporaryFile() as ofile:
            self.dataset.save_as(ofile.name)

    def test_dataset_has_valid_file_meta(self) -> None:
        pydicom.dataset.validate_file_meta(self.dataset.file_meta)

    def test_file_meta_has_information_group_length_computed(self) -> None:
        assert "FileMetaInformationGroupLength" in self.dataset.file_meta
        assert self.dataset.file_meta.FileMetaInformationGroupLength > 0

    def test_mandatory_sop_common(self) -> None:
        assert self.dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
        assert "SOPInstanceUID" in self.dataset

    def test_mandatory_enhanced_equipment_elements(self) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.5.2.html#table_C.7-8b"""
        assert self.dataset.Manufacturer == "pydicom-seg"
        assert (
            self.dataset.ManufacturerModelName
            == "https://github.com/razorx89/pydicom-seg"
        )
        assert self.dataset.DeviceSerialNumber == "0"
        assert self.dataset.SoftwareVersions == __version__

    def test_mandatory_frame_of_reference_elements(self) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.4.html#table_C.7-6"""
        assert "FrameOfReferenceUID" in self.dataset

    def test_mandatory_gernal_series_elements(self) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.3.html#table_C.7-5a"""
        assert self.dataset.Modality == "SEG"
        assert "SeriesInstanceUID" in self.dataset

    def test_mandatory_segmentation_series_elements(self) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html#table_C.8.20-1"""
        assert self.dataset.Modality == "SEG"
        assert self.dataset.SeriesNumber

    def test_mandatory_image_pixel_elements(self) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#table_C.7-11a"""
        assert self.dataset.SamplesPerPixel >= 1
        assert self.dataset.PhotometricInterpretation in ["MONOCHROME1", "MONOCHROME2"]
        assert "Rows" in self.dataset
        assert "Columns" in self.dataset
        assert self.dataset.BitsAllocated in [1, 8, 16]
        assert 0 < self.dataset.BitsStored <= self.dataset.BitsAllocated
        assert self.dataset.HighBit == self.dataset.BitsStored - 1
        assert self.dataset.PixelRepresentation in [0, 1]

    def test_mandatory_and_common_segmentation_image_elements(self) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html#table_C.8.20-2"""
        assert "ImageType" in self.dataset
        assert all(
            [a == b for a, b in zip(self.dataset.ImageType, ["DERIVED", "PRIMARY"])]
        )
        assert self.dataset.InstanceNumber
        assert self.dataset.ContentLabel == "SEGMENTATION"
        assert "ContentCreatorName" in self.dataset
        assert "ContentDescription" in self.dataset
        assert self.dataset.SamplesPerPixel == 1
        assert self.dataset.PhotometricInterpretation == "MONOCHROME2"
        assert self.dataset.PixelRepresentation == 0
        assert self.dataset.LossyImageCompression == "00"
        assert "SegmentSequence" in self.dataset

    def test_mandatory_binary_segmentation_image_elements(self) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html#table_C.8.20-2"""
        assert self.dataset.BitsAllocated == 1
        assert self.dataset.BitsStored == 1
        assert self.dataset.HighBit == 0
        assert self.dataset.SegmentationType == "BINARY"

    @pytest.mark.parametrize("fractional_type", ["PROBABILITY", "OCCUPANCY"])
    def test_mandatory_fractional_segmentation_image_elements(
        self, fractional_type: str
    ) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html#table_C.8.20-2"""
        dataset = SegmentationDataset(
            rows=1,
            columns=1,
            segmentation_type=SegmentationType.FRACTIONAL,
            segmentation_fractional_type=SegmentationFractionalType(fractional_type),
        )
        assert dataset.BitsAllocated == 8
        assert dataset.BitsStored == 8
        assert dataset.HighBit == 7  # Little Endian
        assert dataset.SegmentationType == "FRACTIONAL"
        assert dataset.SegmentationFractionalType == fractional_type
        assert dataset.MaximumFractionalValue == 255

    def test_mandatory_multi_frame_functional_groups_elements(self) -> None:
        """http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.html#table_C.7.6.16-1"""
        assert "SharedFunctionalGroupsSequence" in self.dataset
        assert len(self.dataset.SharedFunctionalGroupsSequence) == 1
        assert "PerFrameFunctionalGroupsSequence" in self.dataset
        assert self.dataset.NumberOfFrames == 0
        assert self.dataset.InstanceNumber
        assert "ContentDate" in self.dataset
        assert "ContentTime" in self.dataset

    def test_timestamps_exist(self) -> None:
        assert "InstanceCreationDate" in self.dataset
        assert "InstanceCreationTime" in self.dataset
        assert self.dataset.InstanceCreationDate == self.dataset.SeriesDate
        assert self.dataset.InstanceCreationTime == self.dataset.SeriesTime
        assert self.dataset.InstanceCreationDate == self.dataset.ContentDate
        assert self.dataset.InstanceCreationTime == self.dataset.ContentTime

    def test_exception_on_invalid_image_dimensions(self) -> None:
        with pytest.raises(ValueError, match=".*must be larger than zero"):
            SegmentationDataset(
                rows=0, columns=0, segmentation_type=SegmentationType.BINARY
            )

    @pytest.mark.parametrize("max_fractional_value", [-1, 0, 256])
    def test_exception_on_invalid_max_fractional_value(
        self, max_fractional_value: int
    ) -> None:
        with pytest.raises(ValueError, match="Invalid maximum fractional value.*"):
            SegmentationDataset(
                rows=1,
                columns=1,
                segmentation_type=SegmentationType.FRACTIONAL,
                max_fractional_value=max_fractional_value,
            )

    def test_exception_when_adding_frame_with_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match=".*expecting 2D image"):
            self.dataset.add_frame(np.zeros((1, 1, 1), dtype=np.uint8), 1)

    def test_exception_when_adding_frame_with_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=".*expecting \\d+x\\d+ images"):
            self.dataset.add_frame(np.zeros((2, 1), dtype=np.uint8), 1)

    @pytest.mark.parametrize(
        "segmentation_type,dtype",
        [
            (SegmentationType.BINARY, np.float32),
            (SegmentationType.FRACTIONAL, np.uint8),
        ],
    )
    def test_exception_when_adding_frame_with_wrong_data_type(
        self, segmentation_type: SegmentationType, dtype: np.dtype
    ) -> None:
        dataset = SegmentationDataset(
            rows=1, columns=1, segmentation_type=segmentation_type
        )
        with pytest.raises(ValueError, match=".*requires.*?data type"):
            dataset.add_frame(np.zeros((1, 1), dtype=dtype), 1)

    def test_adding_frame_increases_number_of_frames(self) -> None:
        old_count = self.dataset.NumberOfFrames
        print(type(old_count))
        self.dataset.add_frame(np.zeros((1, 1), dtype=np.uint8), 1)
        assert self.dataset.NumberOfFrames == old_count + 1

    def test_adding_binary_frame_modifies_pixel_data(self) -> None:
        dataset = SegmentationDataset(
            rows=2, columns=2, segmentation_type=SegmentationType.BINARY
        )
        self.setup_dummy_segment(dataset)
        assert len(dataset.PixelData) == 0

        dataset.add_frame(np.zeros((2, 2), dtype=np.uint8), 1)
        assert len(dataset.PixelData) == 1

        for _ in range(2):
            dataset.add_frame(np.ones((2, 2), dtype=np.uint8), 1)
        assert len(dataset.PixelData) == 2

    def test_adding_fractional_frame_modifies_pixel_data(self) -> None:
        dataset = SegmentationDataset(
            rows=2, columns=2, segmentation_type=SegmentationType.FRACTIONAL
        )
        self.setup_dummy_segment(dataset)
        assert len(dataset.PixelData) == 0

        dataset.add_frame(np.zeros((2, 2), dtype=np.float32), 1)
        assert len(dataset.PixelData) == 4

        for _ in range(2):
            dataset.add_frame(np.ones((2, 2), dtype=np.float32), 1)
        assert len(dataset.PixelData) == 12

    def test_adding_frame_with_reference_creates_referenced_series_sequence(
        self,
    ) -> None:
        assert "ReferencedSeriesSequence" not in self.dataset

        dummy = self.generate_dummy_source_image()

        self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1, [dummy])
        assert "ReferencedSeriesSequence" in self.dataset
        series_sequence = self.dataset.ReferencedSeriesSequence
        assert len(series_sequence) == 1
        assert series_sequence[0].SeriesInstanceUID == dummy.SeriesInstanceUID

        assert "ReferencedInstanceSequence" in series_sequence[0]
        instance_sequence = series_sequence[0].ReferencedInstanceSequence
        assert len(instance_sequence) == 1
        assert instance_sequence[0].ReferencedSOPClassUID == dummy.SOPClassUID
        assert instance_sequence[0].ReferencedSOPInstanceUID == dummy.SOPInstanceUID

    def test_adding_frames_with_different_references_from_same_series(self) -> None:
        dummy1 = self.generate_dummy_source_image()
        dummy2 = self.generate_dummy_source_image()
        dummy2.SeriesInstanceUID = dummy1.SeriesInstanceUID

        self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1, [dummy1])
        self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1, [dummy2])
        series_sequence = self.dataset.ReferencedSeriesSequence
        assert len(series_sequence) == 1
        assert series_sequence[0].SeriesInstanceUID == dummy1.SeriesInstanceUID

        instance_sequence = series_sequence[0].ReferencedInstanceSequence
        assert len(instance_sequence) == 2
        assert instance_sequence[0].ReferencedSOPInstanceUID == dummy1.SOPInstanceUID
        assert instance_sequence[1].ReferencedSOPInstanceUID == dummy2.SOPInstanceUID

    def test_adding_frames_with_different_references_from_different_series(
        self,
    ) -> None:
        dummies = [self.generate_dummy_source_image() for _ in range(2)]

        self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1, [dummies[0]])
        self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1, [dummies[1]])
        series_sequence = self.dataset.ReferencedSeriesSequence
        assert len(series_sequence) == 2
        assert series_sequence[0].SeriesInstanceUID == dummies[0].SeriesInstanceUID
        assert series_sequence[1].SeriesInstanceUID == dummies[1].SeriesInstanceUID

        instance_sequence = series_sequence[0].ReferencedInstanceSequence
        assert len(instance_sequence) == 1
        assert (
            instance_sequence[0].ReferencedSOPInstanceUID == dummies[0].SOPInstanceUID
        )

        instance_sequence = series_sequence[1].ReferencedInstanceSequence
        assert len(instance_sequence) == 1
        assert (
            instance_sequence[0].ReferencedSOPInstanceUID == dummies[1].SOPInstanceUID
        )

    def test_adding_instance_reference_multiple_times(self) -> None:
        dummy = self.generate_dummy_source_image()

        item_added = self.dataset.add_instance_reference(dummy)
        assert item_added
        item_added = self.dataset.add_instance_reference(dummy)
        assert not item_added

        series_sequence = self.dataset.ReferencedSeriesSequence
        assert len(series_sequence) == 1
        assert series_sequence[0].SeriesInstanceUID == dummy.SeriesInstanceUID
        assert len(series_sequence[0].ReferencedInstanceSequence) == 1

    def test_adding_frame_increases_count_of_per_functional_groups_sequence(
        self,
    ) -> None:
        assert len(self.dataset.PerFrameFunctionalGroupsSequence) == 0
        self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1)
        assert len(self.dataset.PerFrameFunctionalGroupsSequence) == 1

    def test_adding_frame_adds_derivation_image_sequence_to_per_frame_functional_group_item(
        self,
    ) -> None:
        frame_item = self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1)
        assert "DerivationImageSequence" in frame_item

        dummy = self.generate_dummy_source_image()

        frame_item = self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1, [dummy])
        assert "SourceImageSequence" in frame_item.DerivationImageSequence[0]
        assert len(frame_item.DerivationImageSequence[0].SourceImageSequence) == 1

    def test_adding_frame_adds_referenced_segment_to_per_frame_functional_group_item(
        self,
    ) -> None:
        frame_item = self.dataset.add_frame(np.zeros((1, 1), np.uint8), 1)
        assert "SegmentIdentificationSequence" in frame_item
        assert len(frame_item.SegmentIdentificationSequence) == 1
        segment_id_item = frame_item.SegmentIdentificationSequence[0]
        assert "ReferencedSegmentNumber" in segment_id_item
        assert segment_id_item.ReferencedSegmentNumber == 1

    def test_exception_on_adding_frame_with_non_existing_segment(self) -> None:
        with pytest.raises(IndexError, match="Segment not found.*"):
            self.dataset.add_frame(np.zeros((1, 1), np.uint8), 2)

    def test_add_dimension_organization(self) -> None:
        assert "DimensionOrganizationSequence" not in self.dataset
        assert "DimensionIndexSequence" not in self.dataset

        seq = DimensionOrganizationSequence()
        seq.add_dimension("ReferencedSegmentNumber", "SegmentIdentificationSequence")
        seq.add_dimension("ImagePositionPatient", "PlanePositionSequence")
        self.dataset.add_dimension_organization(seq)

        assert len(self.dataset.DimensionOrganizationSequence) == 1
        assert len(self.dataset.DimensionIndexSequence) == 2
        assert (
            self.dataset.DimensionIndexSequence[0].DimensionDescriptionLabel
            == "ReferencedSegmentNumber"
        )
        assert (
            self.dataset.DimensionIndexSequence[1].DimensionDescriptionLabel
            == "ImagePositionPatient"
        )

    def test_add_dimension_organization_duplicate(self) -> None:
        seq = DimensionOrganizationSequence()
        seq.add_dimension("ReferencedSegmentNumber", "SegmentIdentificationSequence")
        seq.add_dimension("ImagePositionPatient", "PlanePositionSequence")
        self.dataset.add_dimension_organization(seq)
        with pytest.raises(ValueError, match="Dimension organization with UID.*"):
            self.dataset.add_dimension_organization(seq)

    def test_add_multiple_dimension_organizations(self) -> None:
        for _ in range(2):
            seq = DimensionOrganizationSequence()
            seq.add_dimension(
                "ReferencedSegmentNumber", "SegmentIdentificationSequence"
            )
            seq.add_dimension("ImagePositionPatient", "PlanePositionSequence")
            self.dataset.add_dimension_organization(seq)

        assert len(self.dataset.DimensionOrganizationSequence) == 2
        assert len(self.dataset.DimensionIndexSequence) == 4

    def test_dataset_has_preemble(self) -> None:
        assert self.dataset.get("preamble")
        assert len(self.dataset.preamble) == 128

    def test_dataset_preamble_persisted(self) -> None:
        with tempfile.NamedTemporaryFile() as ofile:
            self.dataset.save_as(ofile.name)
            reloaded_ds = pydicom.dcmread(ofile.name)
        assert reloaded_ds.get("preamble")
        assert len(reloaded_ds.preamble) == 128
        assert self.dataset.preamble == reloaded_ds.preamble
