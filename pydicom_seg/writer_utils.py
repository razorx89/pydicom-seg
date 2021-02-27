import logging
from typing import Set

import pydicom
import SimpleITK as sitk

from pydicom_seg.dicom_utils import sitk_to_dcm_orientation

logger = logging.getLogger(__name__)


def copy_segmentation_template(
    target: pydicom.Dataset,
    template: pydicom.Dataset,
    segments: Set[int],
    skip_missing_segment: bool,
) -> None:
    # Copy mandatory fields
    target.ClinicalTrialSeriesID = template.ClinicalTrialSeriesID
    target.ClinicalTrialTimePointID = template.ClinicalTrialTimePointID
    target.SeriesDescription = template.SeriesDescription
    target.SeriesNumber = template.SeriesNumber

    # Copy optional fields
    if "ClinicalTrialCoordinatingCenterName" in template:
        target.ClinicalTrialCoordinatingCenterName = (
            template.ClinicalTrialCoordinatingCenterName
        )

    target.ContentCreatorName = template.get("ContentCreatorName", "pydicom_seg")
    target.ContentDescription = template.get("ContentDescription", "pydicom_seg")
    target.ContentLabel = template.get("ContentLabel", "pydicom_seg")

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
                raise KeyError(f"Segment {segment_number} was not declared in template")
            logger.warning(
                "Skipping label %d, no meta information declared", segment_number
            )


def import_hierarchy(
    target: pydicom.Dataset,
    reference: pydicom.Dataset,
    import_patient: bool = True,
    import_study: bool = True,
    import_frame_of_reference: bool = False,
    import_series: bool = False,
    import_charset: bool = True,
) -> None:
    """
    Import data elements from a reference DICOM file according to DCMTK
    implementation of DcmIODCommon::importHierarchy.
    """
    tags_to_import = []
    if import_patient:
        # DCMTK IODPatientModule
        tags_to_import.extend(
            ["PatientName", "PatientID", "PatientBirthDate", "PatientSex"]
        )
    if import_study:
        # DCMTK IODGeneralStudyModule
        tags_to_import.extend(
            [
                "StudyInstanceUID",
                "StudyDate",
                "StudyTime",
                "ReferringPhysicianName",
                "StudyID",
                "AccessionNumber",
                "StudyDescription",
                "IssuerOfAccessionNumberSequence",
                "ProcedureCodeSequence",
                "ReasonForPerformedProcedureCodeSequence",
            ]
        )
        # DCMTK IODGeneralEquipmentModule
        tags_to_import.extend(
            [
                "Manufacturer",
                "InstitutionName",
                "InstitutionAddress",
                "StationName",
                "InstitutionalDepartmentName",
                "ManufacturerModelName",
                "DeviceSerialNumber",
                "SoftwareVersions",
            ]
        )
        # DCMTK IODPatientStudyModule
        tags_to_import.extend(
            [
                "AdmittingDiagnosesDescription",
                "PatientAge",
                "PatientSize",
                "PatientWeight",
            ]
        )
    if import_series:
        # DCMTK IODGeneralSeriesModule
        tags_to_import.extend(
            [
                "Modality",
                "SeriesInstanceUID",
                "SeriesNumber",
                "Laterality",
                "SeriesDate",
                "SeriesTime",
                "PerformingPhysicianName",
                "ProtocolName",
                "SeriesDescription",
                "OperatorsName",
                "BodyPartExamined",
                "PatientPosition",
                "ReferencedPerformedProcedureStepSequence",
            ]
        )
    if import_series or import_frame_of_reference:
        # DCMTK IODFoRModule
        tags_to_import.extend(["FrameOfReferenceUID", "PositionReferenceIndicator"])
    if import_charset:
        tags_to_import.append("SpecificCharacterSet")

    for tag_name in tags_to_import:
        if tag_name in target:
            del target[tag_name]
        if tag_name in reference:
            target[tag_name] = reference[tag_name]


def set_shared_functional_groups_sequence(
    target: pydicom.Dataset, segmentation: sitk.Image
) -> None:
    spacing = segmentation.GetSpacing()

    dataset = pydicom.Dataset()
    dataset.PixelMeasuresSequence = [pydicom.Dataset()]
    dataset.PixelMeasuresSequence[0].PixelSpacing = [f"{x:e}" for x in spacing[:2]]
    dataset.PixelMeasuresSequence[0].SliceThickness = f"{spacing[2]:e}"
    dataset.PixelMeasuresSequence[0].SpacingBetweenSlices = f"{spacing[2]:e}"
    dataset.PlaneOrientationSequence = [pydicom.Dataset()]
    dataset.PlaneOrientationSequence[
        0
    ].ImageOrientationPatient = sitk_to_dcm_orientation(segmentation)

    target.SharedFunctionalGroupsSequence = pydicom.Sequence([dataset])
