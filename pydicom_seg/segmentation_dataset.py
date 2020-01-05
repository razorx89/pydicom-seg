from datetime import datetime
import enum
from typing import List

import numpy as np
import pydicom
from pydicom._storage_sopclass_uids import SegmentationStorage

from pydicom_seg import __version__
from pydicom_seg.dicom_utils import CodeSequence, DimensionOrganizationSequence


class SegmentationFractionalType(enum.Enum):
    """Possible values for DICOM tag (0x0062, 0x0010)"""
    PROBABILITY = 'PROBABILITY'
    OCCUPANCY = 'OCCUPANCY'


class SegmentationType(enum.Enum):
    """Possible values for DICOM tag (0x0062, 0x0001)"""
    BINARY = 'BINARY'
    FRACTIONAL = 'FRACTIONAL'


class SegmentationDataset(pydicom.Dataset):
    def __init__(self, *,
                 rows: int,
                 columns: int,
                 segmentation_type: SegmentationType,
                 segmentation_fractional_type: SegmentationFractionalType = SegmentationFractionalType.PROBABILITY,
                 max_fractional_value: int = 255):
        super().__init__()

        self._frames = []

        self.SpecificCharacterSet = 'ISO_IR 100'
        self.SOPClassUID = SegmentationStorage
        self.SOPInstanceUID = pydicom.uid.generate_uid()
        self._set_file_meta()

        # General Series module
        self.Modality = 'SEG'
        self.SeriesInstanceUID = pydicom.uid.generate_uid()
        self.SeriesNumber = 1

        # Generate SOP and Series and General Image timestamps
        timestamp = datetime.now()
        self.InstanceCreationDate = timestamp.strftime('%Y%m%d')
        self.InstanceCreationTime = timestamp.strftime('%H%M%S.%f')
        self.SeriesDate = self.InstanceCreationDate
        self.SeriesTime = self.InstanceCreationTime
        self.ContentDate = self.InstanceCreationDate
        self.ContentTime = self.InstanceCreationTime

        # Frame of Reference module
        self.FrameOfReferenceUID = pydicom.uid.generate_uid()

        # Enhanced General Equipment module
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.5.2.html#table_C.7-8b
        self.Manufacturer = 'pydicom-seg'
        self.ManufacturerModelName = 'git@github.com/razorx89/pydicom-seg.git'
        self.DeviceSerialNumber = '0'
        self.SoftwareVersions = __version__

        # Image Pixel module
        if rows <= 0 or columns <= 0:
            raise ValueError('Rows and columns must be larger than zero')
        self.Rows = rows
        self.Columns = columns
        self.PixelData = b''

        # Segmentation Image module
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html#table_C.8.20-2
        self.ImageType = ['DERIVED', 'PRIMARY']
        self.InstanceNumber = '1'
        self.ContentLabel = 'SEGMENTATION'
        self.ContentDescription = ''
        self.ContentCreatorName = ''
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = 'MONOCHROME2'
        self.PixelRepresentation = 0
        self.LossyImageCompression = '00'
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
                raise ValueError('Invalid maximum fractional value for 8-bit unsigned int data')
            self.MaximumFractionalValue = max_fractional_value
            self.BitsAllocated = 8
            self.BitsStored = 8
            self.HighBit = 7

        # Multi-frame Functional Groups module
        self.SharedFunctionalGroupsSequence = pydicom.Sequence([pydicom.Dataset()])
        self.PerFrameFunctionalGroupsSequence = pydicom.Sequence()
        self.NumberOfFrames = 0

    def _set_file_meta(self):
        self.file_meta = pydicom.Dataset()
        self.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        self.is_little_endian = True  # TODO Remove with pydicom 1.4
        self.is_implicit_VR = False  # TODO Remove with pydicom 1.4
        self.fix_meta_info()

    def add_dimension_organization(self, dim_organization: DimensionOrganizationSequence):
        if 'DimensionOrganizationSequence' not in self:
            self.DimensionOrganizationSequence = pydicom.Sequence()
            self.DimensionIndexSequence = pydicom.Sequence()

        for item in self.DimensionOrganizationSequence:
            if item.DimensionOrganizationUID == dim_organization[0].DimensionOrganizationUID:
                raise ValueError('Dimension organization with UID '
                    f'{item.DimensionOrganizationUID} already exists')
        
        item = pydicom.Dataset()
        item.DimensionOrganizationUID = dim_organization[0].DimensionOrganizationUID
        self.DimensionOrganizationSequence.append(item)
        self.DimensionIndexSequence.extend(dim_organization)

    def add_frame(self,
                  data: np.ndarray,
                  referenced_segment: int,
                  referenced_images: List[pydicom.Dataset] = None
                  ) -> pydicom.Dataset:
        referenced_images = referenced_images or []

        if len(data.shape) != 2:
            raise ValueError('Invalid frame data shape, expecting 2D images')

        if data.shape[0] != self.Rows or data.shape[1] != self.Columns:
            raise ValueError(f'Invalid frame data shape, expecting {self.Rows}x{self.Columns} images')

        # TODO Optimize packing/unpacking/concatenation by using a io.BytesIO
        # TODO stream and track free bits in the last byte
        if self.SegmentationType == SegmentationType.BINARY.value:
            if not np.issubdtype(data.dtype, np.integer):
                raise ValueError('Binary segmentation data requires an integer data type')
            data = np.greater(data, 0, dtype=np.uint8)

            self._frames.append(data)
            self.PixelData = np.packbits(
                np.ravel(self._frames),
                bitorder='little'
            ).tobytes()
        else:
            if not np.issubdtype(data.dtype, np.floating):
                raise ValueError('Fractional segmentation data requires a floating point data type')
            data = np.clip(data, 0.0, 1.0)
            data *= self.MaximumFractionalValue
            data = data.astype(np.uint8)

            self._frames.append(data.ravel())
            self.PixelData = np.concatenate(self._frames).tobytes()

        # A frame was added to the dataset
        self.NumberOfFrames += 1

        # Update (0x5200,0x9230) PerFunctionalGroupsSequence
        frame_fg_item = pydicom.Dataset()
        if referenced_segment not in [x.SegmentNumber for x in self.SegmentSequence]:
            raise IndexError('Segment not found in SegmentSequence')
        frame_fg_item.SegmentIdentificationSequence = pydicom.Sequence([pydicom.Dataset()])
        frame_fg_item.SegmentIdentificationSequence[0].ReferencedSegmentNumber = referenced_segment
        if referenced_images:
            frame_fg_item.SourceImageSequence = pydicom.Sequence()
        for referenced_image in referenced_images:
            # Update (0x0008,0x1115) ReferencedSeriesSequence for each referenced image
            self.add_instance_reference(referenced_image)

            # Add referenced image to SourceImageSequence
            ref = pydicom.Dataset()
            ref.ReferencedSOPClassUID = referenced_image.SOPClassUID
            ref.ReferencedSOPInstanceUID = referenced_image.SOPInstanceUID
            ref.PurposeOfReferenceCodeSequence = CodeSequence('121322', 'DCM', 'Segmentation')
            frame_fg_item.SourceImageSequence.append(ref)
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
        if 'ReferencedSeriesSequence' not in self:
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
