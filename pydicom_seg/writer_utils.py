import logging
from typing import List

import numpy as np
import pydicom
import SimpleITK as sitk


logger = logging.getLogger(__name__)


def copy_segmentation_template(target: pydicom.Dataset,
                               template: pydicom.Dataset,
                               segments: List[int],
                               skip_missing_segment: bool):
    # Copy mandatory fields
    target.ClinicalTrialSeriesID = template.ClinicalTrialSeriesID
    target.ClinicalTrialTimePointID = template.ClinicalTrialTimePointID
    target.ContentCreatorName = template.ContentCreatorName
    target.SeriesDescription = template.SeriesDescription
    target.SeriesNumber = template.SeriesNumber

    # Copy optional fields
    if 'ClinicalTrialCoordinatingCenterName' in template:
        target.ClinicalTrialCoordinatingCenterName = template.ClinicalTrialCoordinatingCenterName

    target.ContentCreatorName = 'pydicom_seg'
    target.ContentDescription = template.get('ContentDescription', 'pydicom_seg')
    target.ContentLabel = template.get('ContentLabel', 'pydicom_seg')

    # Copy segment information
    target.SegmentSequence = []
    for segment_number in segments:
        if segment_number == 0:
            continue

        for segment in template.SegmentSequence:
            if segment.SegmentNumber == segment_number:
                target.SegmentSequence.append(segment)
                break
        else:
            if not skip_missing_segment:
                raise KeyError(f'Segment {segment_number} was not declared in template')
            logger.warning('Skipping label %d, no meta information declared', segment_number)


def create_code_sequence(code_value: str,
                         coding_scheme_designator: str,
                         code_meaning: str) -> pydicom.Sequence:
    dataset = pydicom.Dataset()
    dataset.CodeValue = code_value
    dataset.CodingSchemeDesignator = coding_scheme_designator
    dataset.CodeMeaning = code_meaning
    return pydicom.Sequence([dataset])


def create_referenced_source_image_sequence(referenced_images: List[pydicom.Dataset]) -> pydicom.Sequence:
    result = []
    for referenced_image in referenced_images:
        dataset = pydicom.Dataset()
        dataset.ReferencedSOPClassUID = referenced_image.SOPClassUID
        dataset.ReferencedSOPInstanceUID = referenced_image.SOPInstanceUID
        dataset.PurposeOfReferenceCodeSequence = create_code_sequence(
            '121322', 'DCM', 'Segmentation'
        )
        result.append(dataset)
    return pydicom.Sequence(result)


def import_hierarchy(target: pydicom.Dataset,
                     reference: pydicom.Dataset,
                     import_patient: bool = True,
                     import_study: bool = True,
                     import_frame_of_reference: bool = False,
                     import_series: bool = False,
                     import_charset: bool = True):
    """
    Import data elements from a reference DICOM file according to DCMTK
    implementation of DcmIODCommon::importHierarchy.
    """
    tags_to_import = []
    if import_patient:
        # DCMTK IODPatientModule
        tags_to_import.extend([
            'PatientName',
            'PatientID',
            'PatientBirthDate',
            'PatientSex'
        ])
    if import_study:
        # DCMTK IODGeneralStudyModule
        tags_to_import.extend([
            'StudyInstanceUID',
            'StudyDate',
            'StudyTime',
            'ReferringPhysicianName',
            'StudyID',
            'AccessionNumber',
            'StudyDescription',
            'IssuerOfAccessionNumberSequence',
            'ProcedureCodeSequence',
            'ReasonForPerformedProcedureCodeSequence'
        ])
        # DCMTK IODGeneralEquipmentModule
        tags_to_import.extend([
            'Manufacturer',
            'InstitutionName',
            'InstitutionAddress',
            'StationName',
            'InstitutionalDepartmentName',
            'ManufacturerModelName',
            'DeviceSerialNumber',
            'SoftwareVersions'
        ])
        # DCMTK IODPatientStudyModule
        tags_to_import.extend([
            'AdmittingDiagnosesDescription',
            'PatientAge',
            'PatientSize',
            'PatientWeight'
        ])
    if import_series:
        # DCMTK IODGeneralSeriesModule
        tags_to_import.extend([
            'Modality',
            'SeriesInstanceUID',
            'SeriesNumber',
            'Laterality',
            'SeriesDate',
            'SeriesTime',
            'PerformingPhysicianName',
            'ProtocolName',
            'SeriesDescription',
            'OperatorsName',
            'BodyPartExamined',
            'PatientPosition',
            'ReferencedPerformedProcedureStepSequence'
        ])
    if import_series or import_frame_of_reference:
        # DCMTK IODFoRModule
        tags_to_import.extend([
            'FrameOfReferenceUID',
            'PositionReferenceIndicator'
        ])
    if import_charset:
        tags_to_import.append('SpecificCharacterSet')

    for tag_name in tags_to_import:
        if tag_name in target:
            del target[tag_name]
        if tag_name in reference:
            target[tag_name] = reference[tag_name]


def set_binary_segmentation(target: pydicom.Dataset):
    target.BitsAllocated = 1
    target.BitsStored = 1
    target.HighBit = 0
    target.LossyImageCompression = '00'
    target.PhotometricInterpretation = 'MONOCHROME2'
    target.PixelRepresentation = 0
    target.SamplesPerPixel = 1
    target.SegmentationType = 'BINARY'


def set_default_dimension_organization(target: pydicom.Dataset):
    dim_uid = pydicom.uid.generate_uid()
    index0 = pydicom.Dataset()
    index0.DimensionOrganizationUID = dim_uid

    # First index
    index1 = pydicom.Dataset()
    index1.DimensionOrganizationUID = dim_uid
    index1.DimensionIndexPointer = pydicom.tag.Tag(
        pydicom.datadict.tag_for_keyword('ReferencedSegmentNumber')
    )
    index1.FunctionalGroupPointer = pydicom.tag.Tag(
        pydicom.datadict.tag_for_keyword('SegmentIdentificationSequence')
    )
    index1.DimensionDescriptionLabel = 'ReferencedSegmentNumber'

    # Second index
    index2 = pydicom.Dataset()
    index2.DimensionOrganizationUID = dim_uid
    index2.DimensionIndexPointer = pydicom.tag.Tag(
        pydicom.datadict.tag_for_keyword('ImagePositionPatient')
    )
    index2.FunctionalGroupPointer = pydicom.tag.Tag(
        pydicom.datadict.tag_for_keyword('PlanePositionSequence')
    )
    index2.DimensionDescriptionLabel = 'ImagePositionPatient'

    target.DimensionIndexSequence = pydicom.Sequence([index0, index1, index2])


def set_shared_functional_groups_sequence(target: pydicom.Dataset, segmentation: sitk.Image):
    spacing = segmentation.GetSpacing()

    dataset = pydicom.Dataset()
    dataset.PixelMeasuresSequence = [pydicom.Dataset()]
    dataset.PixelMeasuresSequence[0].PixelSpacing = [f'{x:e}' for x in spacing[:2]]
    dataset.PixelMeasuresSequence[0].SliceThickness = f'{spacing[2]:e}'
    dataset.PixelMeasuresSequence[0].SpacingBetweenSlices = f'{spacing[2]:e}'
    dataset.PlaneOrientationSequence = [pydicom.Dataset()]
    dataset.PlaneOrientationSequence[0].ImageOrientationPatient = [
        f'{x:e}' for x in np.ravel(segmentation.GetDirection())[:6]
    ]

    target.SharedFunctionalGroupsSequence = pydicom.Sequence([dataset])
