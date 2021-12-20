import enum
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pydicom
from pydicom._storage_sopclass_uids import SegmentationStorage

from pydicom_seg import __version__, writer_utils
from pydicom_seg.dicom_utils import CodeSequence, DimensionOrganizationSequence

logger = logging.getLogger(__name__)


class SegmentationFractionalType(enum.Enum):
    """Possible values for DICOM tag (0x0062, 0x0010)"""

    PROBABILITY = "PROBABILITY"
    OCCUPANCY = "OCCUPANCY"


class SegmentationType(enum.Enum):
    """Possible values for DICOM tag (0x0062, 0x0001)"""

    BINARY = "BINARY"
    FRACTIONAL = "FRACTIONAL"


class SegmentsOverlap(enum.Enum):
    """Possible values for DICOM tag (0x0062, 0x0013)"""

    YES = "YES"
    UNDEFINED = "UNDEFINED"
    NO = "NO"


class SegmentationDataset(pydicom.Dataset):
    """Specialized dataset for Segmentation IOD.

    This dataset class initializes most of the mandatory DICOM elements for
    a valid DICOM-SEG file. Patient and study level elements are not
    initialized and are expected to be imported from other DICOM datasets.
    Additionally, it provides utility functions for manipulating various
    attributes and ensures consistency, for instance when adding frame data.

    If the segmentation type is specified as `BINARY`, added frame data is
    expected to be of integer data type and will be encoded as one bit per
    pixel. If the segmentation type is specified as `FRACTIONAL`, then
    frame data is expected to be of floating point data type and will be
    clipped to the value range `[0.0, 1.0]`. Afterwards, the data is
    quantized into 8-bit with `max_fractional_value` being equal to `1.0`.

    Example:
        ::

            ds = SegmentationDataset(
                rows=512,
                columns=512,
                segmentation_type=SegmentationType.BINARY
            )
            # Further configure the dataset, e.g. segment information
            frame_fg_item = ds.add_frame(
                data=np.ones((512, 512), dtype=np.uint8),
                referenced_segment=1,
            )
            # Further configure the PerFrameFunctionalGroupSequence item

    Args:
        rows: Number of rows in a frame/image (y-axis)
        columns: Number of columns in a frame/image (x-axis)
        segmentation_type: Either `SegmentationType.BINARY` or
            `SegmentationType.FRACTIONAL` depending on the data to encode.
        segmentation_fractional_type: If `segmentation_type == SegmentationType.FRACTIONAL`,
            then the fractional type indicates the semantic meaning. Can be
            either `PROBABILITY` or `OCCUPANCY`.
        max_fractional_value: Fractional data is expected to be within
            `[0.0, 1.0]` and will be rescaled to `[0, max_fractional_value]`.
    """

    def __init__(
        self,
        *,
        rows: int,
        columns: int,
        segmentation_type: SegmentationType,
        segmentation_fractional_type: SegmentationFractionalType = SegmentationFractionalType.PROBABILITY,
        reference_dicom: Optional[pydicom.Dataset] = None,
        max_fractional_value: int = 255,
    ):
        super().__init__()

        self._frames: List[np.ndarray] = []
        if reference_dicom:
            writer_utils.import_hierarchy(
                target=self,
                reference=reference_dicom,
                import_frame_of_reference=True,
                import_series=False,
            )
        else:
            logger.warning(
                "No source images provided, cannot import patient "
                "and study level information."
            )

        self.preamble = b"\0" * 128
        self.SpecificCharacterSet = "ISO_IR 100"
        self.SOPClassUID = SegmentationStorage
        self.SOPInstanceUID = pydicom.uid.generate_uid()
        self._init_file_meta()

        # General Series module
        self.Modality = "SEG"
        self.SeriesInstanceUID = pydicom.uid.generate_uid()
        self.SeriesNumber = 1

        # Generate SOP and Series and General Image timestamps
        timestamp = datetime.now()
        self.InstanceCreationDate = timestamp.strftime("%Y%m%d")
        self.InstanceCreationTime = timestamp.strftime("%H%M%S.%f")
        self.SeriesDate = self.InstanceCreationDate
        self.SeriesTime = self.InstanceCreationTime
        self.ContentDate = self.InstanceCreationDate
        self.ContentTime = self.InstanceCreationTime

        # Frame of Reference module
        self.FrameOfReferenceUID = pydicom.uid.generate_uid()

        # Enhanced General Equipment module
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.5.2.html#table_C.7-8b
        self.Manufacturer = "pydicom-seg"
        self.ManufacturerModelName = "https://github.com/razorx89/pydicom-seg"
        self.DeviceSerialNumber = "0"
        self.SoftwareVersions = __version__

        # Image Pixel module
        if rows <= 0 or columns <= 0:
            raise ValueError("Rows and columns must be larger than zero")
        self.Rows = rows
        self.Columns = columns
        self.PixelData = b""

        # Segmentation Image module
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html#table_C.8.20-2
        self.ImageType = ["DERIVED", "PRIMARY"]
        self.InstanceNumber = "1"
        self.ContentLabel = "SEGMENTATION"
        self.ContentDescription = ""
        self.ContentCreatorName = ""
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = "MONOCHROME2"
        self.PixelRepresentation = 0
        self.LossyImageCompression = "00"
        self.SegmentSequence = pydicom.Sequence()

        self.SegmentationType = segmentation_type.value
        if segmentation_type == SegmentationType.BINARY:
            # Binary segmentations are always bit-packed
            self.BitsAllocated = 1
            self.BitsStored = 1
            self.HighBit = 0
        else:
            # Fractional segmentations are always 8-bit unsigned
            self.SegmentationFractionalType = segmentation_fractional_type.value
            if max_fractional_value < 1 or max_fractional_value > 255:
                raise ValueError(
                    "Invalid maximum fractional value for 8-bit unsigned int data"
                )
            self.MaximumFractionalValue = max_fractional_value
            self.BitsAllocated = 8
            self.BitsStored = 8
            self.HighBit = 7

        # Multi-frame Functional Groups module
        self.SharedFunctionalGroupsSequence = pydicom.Sequence([pydicom.Dataset()])
        self.PerFrameFunctionalGroupsSequence = pydicom.Sequence()
        self.NumberOfFrames = 0

    def _init_file_meta(self) -> None:
        if pydicom.__version_info__[0] == "1":
            self.file_meta = pydicom.Dataset()
        else:
            self.file_meta = pydicom.dataset.FileMetaDataset()
        self.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        self.file_meta.MediaStorageSOPInstanceUID = self.SOPInstanceUID
        self.file_meta.MediaStorageSOPClassUID = self.SOPClassUID
        pydicom.dataset.validate_file_meta(self.file_meta)

        # Fix missing FileMetaInformationGroupLength. It is added by `pydicom` when saving with
        # `write_as_original=False`, but this can be a dangerous pitfall if not done correctly
        if "FileMetaInformationGroupLength" not in self.file_meta:
            # See: https://github.com/pydicom/pydicom/blob/e8de9d31fc97e1162441adf4bd2742b82149ce18/pydicom/filewriter.py#L645-L736
            buffer = pydicom.filewriter.DicomBytesIO()
            buffer.is_little_endian = True
            buffer.is_implicit_VR = False
            pydicom.filewriter.write_dataset(buffer, self.file_meta)
            self.file_meta.FileMetaInformationGroupLength = buffer.tell()

    def update_pixel_data(self) -> None:
        """Replaces the pixel data using the list of frames.

        This methods replaces the (0x7fe0, 0x0010) Pixel Data by using the
        prestored frames. If this method is called after adding every frame,
        it can be slow. To speed up, disable it and call this method after
        finishing adding frames.
        """
        raw_pixel_data = np.concatenate(self._frames)
        if self.SegmentationType == SegmentationType.BINARY.value:
            self.PixelData = np.packbits(raw_pixel_data, bitorder="little").tobytes()
        else:
            self.PixelData = raw_pixel_data.tobytes()

    def add_dimension_organization(
        self, dim_organization: DimensionOrganizationSequence
    ) -> None:
        """Adds a dimension organization sequence to the dataset.

        This methods registers the (0x0020, 0x9164) DimensionOrganizationUID
        and appends all items from the sequence to (0x0020, 0x9222)
        DimensionIndexSequence.

        Args:
            dim_organization: A `DimensionOrganizationSequence` with one or
                more dimension items configured.
        """
        if "DimensionOrganizationSequence" not in self:
            self.DimensionOrganizationSequence = pydicom.Sequence()
            self.DimensionIndexSequence = pydicom.Sequence()

        for item in self.DimensionOrganizationSequence:
            if (
                item.DimensionOrganizationUID
                == dim_organization[0].DimensionOrganizationUID
            ):
                raise ValueError(
                    "Dimension organization with UID "
                    f"{item.DimensionOrganizationUID} already exists"
                )

        item = pydicom.Dataset()
        item.DimensionOrganizationUID = dim_organization[0].DimensionOrganizationUID
        self.DimensionOrganizationSequence.append(item)
        self.DimensionIndexSequence.extend(dim_organization)

    def add_frame(
        self,
        data: np.ndarray,
        referenced_segment: int,
        referenced_images: List[pydicom.Dataset] = None,
        update_pixel_data: bool = True,
    ) -> pydicom.Dataset:
        """Adds a frame to the dataset.

        Adds the frame data to the PixelData element after encoding the data
        into the correct format for the configured segmentation type.
        Each referenced image is also registered in (0x0008,0x1115)
        ReferencedSeriesSequence.

        Args:
            data: A `np.ndarray` with integer data type for binary
                or floating point data type for fractional segmentations.
            referenced_segment: An integer number for the segment to which
                this frame belongs. The segment must exist in (0x0062, 0x0002)
                SegmentSequence.
            referenced_images: A list of `pydicom.Dataset` source images, which
                map to the frame.
            update_pixel_data: a boolean indicating whether the pixeldata
                should be updated right after adding the frame. Otherwise
                update_pixel_data() has to be called after adding all frames.
                Calling update_pixel_data() once at the end increases speed.

        Returns:
            A `pydicom.Dataset` with pre-initialized values for an item in
            (0x5200,0x9230) PerFrameFunctionalGroupsSequence, which needs
            further configuration based on writer dependent strategy.
        """
        referenced_images = referenced_images or []

        if len(data.shape) != 2:
            raise ValueError("Invalid frame data shape, expecting 2D images")

        if data.shape[0] != self.Rows or data.shape[1] != self.Columns:
            raise ValueError(
                f"Invalid frame data shape, expecting {self.Rows}x{self.Columns} images"
            )

        # TODO Optimize packing/unpacking/concatenation by using a io.BytesIO
        # TODO stream and track free bits in the last byte
        if self.SegmentationType == SegmentationType.BINARY.value:
            if not np.issubdtype(data.dtype, np.integer):
                raise ValueError(
                    "Binary segmentation data requires an integer data type"
                )
            data = np.greater(data, 0, dtype=np.uint8)

        else:
            if not np.issubdtype(data.dtype, np.floating):
                raise ValueError(
                    "Fractional segmentation data requires a floating point data type"
                )
            data = np.clip(data, 0.0, 1.0)
            data *= self.MaximumFractionalValue
            data = data.astype(np.uint8)

        self._frames.append(data.ravel())

        # Update PixelData if required, otherwise it must be done after adding all frames
        if update_pixel_data:
            self.update_pixel_data()

        # A frame was added to the dataset
        self.NumberOfFrames += 1

        # Update (0x5200,0x9230) PerFunctionalGroupsSequence
        frame_fg_item = pydicom.Dataset()
        if referenced_segment not in [x.SegmentNumber for x in self.SegmentSequence]:
            raise IndexError("Segment not found in SegmentSequence")
        frame_fg_item.SegmentIdentificationSequence = pydicom.Sequence(
            [pydicom.Dataset()]
        )
        frame_fg_item.SegmentIdentificationSequence[
            0
        ].ReferencedSegmentNumber = referenced_segment

        # Each frame requires references to the original DICOM files
        derivation_image = pydicom.Dataset()
        derivation_image.SourceImageSequence = pydicom.Sequence()

        for referenced_image in referenced_images:
            # Update (0x0008,0x1115) ReferencedSeriesSequence for each referenced image
            self.add_instance_reference(referenced_image)

            # Add referenced image to SourceImageSequence
            ref = pydicom.Dataset()
            ref.ReferencedSOPClassUID = referenced_image.SOPClassUID
            ref.ReferencedSOPInstanceUID = referenced_image.SOPInstanceUID
            ref.PurposeOfReferenceCodeSequence = CodeSequence(
                "121322", "DCM", "Source image for image processing operation"
            )
            derivation_image.SourceImageSequence.append(ref)
        derivation_image.DerivationCodeSequence = CodeSequence(
            "113076", "DCM", "Segmentation"
        )
        frame_fg_item.DerivationImageSequence = pydicom.Sequence([derivation_image])
        self.PerFrameFunctionalGroupsSequence.append(frame_fg_item)

        return frame_fg_item

    def add_instance_reference(self, dataset: pydicom.Dataset) -> bool:
        """Adds an instance to the (0x0008,0x1115) ReferencedSeriesSequence.

        Args:
            dataset: A `pydicom.Dataset` DICOM image which should be added
                to the segmentation dataset.

        Returns:
            Returns `True`, if `dataset` was added as a reference.
        """
        if "ReferencedSeriesSequence" not in self:
            self.ReferencedSeriesSequence = pydicom.Sequence()

        for series_item in self.ReferencedSeriesSequence:
            if series_item.SeriesInstanceUID != dataset.SeriesInstanceUID:
                continue

            for instance_item in series_item.ReferencedInstanceSequence:
                if instance_item.ReferencedSOPInstanceUID == dataset.SOPInstanceUID:
                    return False

            # Series found, but instance is missing
            break
        else:
            # Series not yet referenced, create a new series item
            series_item = pydicom.Dataset()
            series_item.SeriesInstanceUID = dataset.SeriesInstanceUID
            series_item.ReferencedInstanceSequence = pydicom.Sequence([])
            self.ReferencedSeriesSequence.append(series_item)

        # Instance not yet referenced, create a new instance item
        instance_item = pydicom.Dataset()
        instance_item.ReferencedSOPClassUID = dataset.SOPClassUID
        instance_item.ReferencedSOPInstanceUID = dataset.SOPInstanceUID
        series_item.ReferencedInstanceSequence.append(instance_item)

        return True
