import json
import os
from typing import List, Union

import jsonschema
import pydicom


def _create_validator() -> jsonschema.Draft4Validator:
    """Create a JSON validator instance from dcmqi schema files.

    In order to allow offline usage, the required schemas a pre-loaded from the
    dcmqi repository located at `pydicom_seg/externals/dcmqi`.

    Returns:
        A `jsonschema.Draft4Validator` with a pre-loaded schema store.
    """
    # Load both common and segmentation schema files
    schemas_dir = os.path.join(
        os.path.dirname(__file__), "externals", "dcmqi", "doc", "schemas"
    )
    seg_schema_path = os.path.join(schemas_dir, "seg-schema.json")
    with open(seg_schema_path) as ifile:
        seg_schema = json.load(ifile)
    with open(os.path.join(schemas_dir, "common-schema.json")) as ifile:
        common_schema = json.load(ifile)

    # Create validator with pre-loaded schema store
    return jsonschema.Draft4Validator(
        seg_schema,
        resolver=jsonschema.RefResolver(
            base_uri="file://" + seg_schema_path,
            referrer=seg_schema,
            store={seg_schema["id"]: seg_schema, common_schema["id"]: common_schema},
        ),
    )


def _create_code_sequence(data: dict) -> pydicom.Sequence:
    """Helper function for creating a DICOM sequence from JSON attributes.

    Returns:
        A `pydicom.Sequence` with a single `pydicom.Dataset` item containing
        all attributes from the JSON document.
    """
    dataset = pydicom.Dataset()
    for key in data:
        dataset.__setattr__(key, data[key])
    return pydicom.Sequence([dataset])


def _create_segment_dataset(data: dict) -> pydicom.Dataset:
    """Helper function for creating an item for SegmentSequence.

    Args:
        data: A `dict` with information about a single segment.

    Returns:
        A `pydicom.Dataset` containing all required and optional information
        about an annotated segment.
    """
    dataset = pydicom.Dataset()

    # Mandatory fields
    dataset.SegmentAlgorithmName = data.get("SegmentAlgorithmName", "")
    dataset.SegmentAlgorithmType = data.get("SegmentAlgorithmType", "SEMIAUTOMATIC")
    dataset.SegmentNumber = data["labelID"]
    dataset.SegmentedPropertyCategoryCodeSequence = _create_code_sequence(
        data["SegmentedPropertyCategoryCodeSequence"]
    )
    dataset.SegmentedPropertyTypeCodeSequence = _create_code_sequence(
        data["SegmentedPropertyTypeCodeSequence"]
    )

    # Optional fields
    optional_code_sequences = [
        "SegmentedPropertyTypeModifierCodeSequence",
        "AnatomicRegionSequence",
        "AnatomicRegionModifierSequence",
    ]
    for code_sequence in optional_code_sequences:
        if code_sequence in data:
            dataset.__setattr__(
                code_sequence, _create_code_sequence(data[code_sequence])
            )

    optional_tags_no_default = [
        "SegmentDescription",
        "SegmentLabel",
        "TrackingIdentifier",
        "TrackingUniqueIdentifier",
    ]
    for tag_name in optional_tags_no_default:
        if tag_name in data:
            dataset.__setattr__(tag_name, data[tag_name])

    if "RecommendedDisplayCIELabValue" in data:
        dataset.RecommendedDisplayCIELabValue = data["RecommendedDisplayCIELabValue"]
    elif "recommendedDisplayRGBValue" in data:
        dataset.RecommendedDisplayCIELabValue = rgb_to_cielab(
            data["recommendedDisplayRGBValue"]
        )

    return dataset


def from_dcmqi_metainfo(metainfo: Union[dict, str]) -> pydicom.Dataset:
    """Converts a `metainfo.json` file from the dcmqi project to a
    `pydicom.Dataset` with the matching DICOM data elements set from JSON.

    Those JSON files can be easilly created using the segmentation editor
    tool from QIICR/dcmqi:
    http://qiicr.org/dcmqi/#/seg
    When converting the JSON to a DICOM dataset, the validity of the provided
    JSON document is ensured using the official JSON schema files from the
    dcmqi project.

    Args:
        metainfo: Either a `str` for a file path to read from or a `dict`
            with the JSON already imported or constructed in source code.

    Returns:
        A `pydicom.Dataset` containg all values from the JSON document and
        some defaults if the elements were not available.
    """
    # Add convienence loader of JSON dictionary
    if isinstance(metainfo, str):
        with open(metainfo) as ifile:
            metainfo = json.load(ifile)
    assert isinstance(metainfo, dict)

    # Validate dictionary against dcmqi JSON schemas
    validator = _create_validator()
    if not validator.is_valid(metainfo):
        raise NotImplementedError()

    # Create dataset from provided JSON
    dataset = pydicom.Dataset()
    tags_with_defaults = [
        ("BodyPartExamined", ""),
        ("ClinicalTrialCoordinatingCenterName", ""),
        ("ClinicalTrialSeriesID", "Session1"),
        ("ClinicalTrialTimePointID", "1"),
        ("ContentCreatorName", "Reader1"),
        ("ContentDescription", "Image segmentation"),
        ("ContentLabel", "SEGMENTATION"),
        ("InstanceNumber", "1"),
        ("SeriesDescription", "Segmentation"),
        ("SeriesNumber", "300"),
    ]

    for tag_name, default_value in tags_with_defaults:
        dataset.__setattr__(tag_name, metainfo.get(tag_name, default_value))

    if len(metainfo["segmentAttributes"]) > 1:
        raise ValueError(
            "Only metainfo.json files written for single-file input are supported"
        )

    dataset.SegmentSequence = pydicom.Sequence(
        [_create_segment_dataset(x) for x in metainfo["segmentAttributes"][0]]
    )

    return dataset


def rgb_to_cielab(rgb: List[int]) -> List[int]:
    """
    Convert from RGB to CIELab color space.

    Arguments:
        rgb: Iterable of length 3 with integer values between 0 and 255

    Returns:
        A list with 3 integer scaled values of CIELab.

    References:
        - https://github.com/QIICR/dcmqi/blob/0c101b702f12a86cc142cb000a074fbd341f8784/libsrc/Helper.cpp#L173
        - https://github.com/QIICR/dcmqi/blob/0c101b702f12a86cc142cb000a074fbd341f8784/libsrc/Helper.cpp#L256
        - https://github.com/QIICR/dcmqi/blob/0c101b702f12a86cc142cb000a074fbd341f8784/libsrc/Helper.cpp#L336
    """
    assert len(rgb) == 3

    # RGB -> CIEXYZ (Ref 1)
    tmp = tuple(min(max(x / 255.0, 0.0), 1.0) for x in rgb)
    tmp = tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in tmp)
    tmp = tuple(x * 100 for x in tmp)
    tmp = (
        0.4124564 * tmp[0] + 0.3575761 * tmp[1] + 0.1804375 * tmp[2],
        0.2126729 * tmp[0] + 0.7151522 * tmp[1] + 0.0721750 * tmp[2],
        0.0193339 * tmp[0] + 0.1191920 * tmp[1] + 0.9503041 * tmp[2],
    )

    # CIEXYZ -> CIELab (Ref 2)
    tmp = tuple(x / y for x, y in zip(tmp, (95.047, 100, 108.883)))
    tmp = tuple(x ** (1 / 3) if x > 0.008856 else (7.787 * x) + (16 / 116) for x in tmp)
    tmp = ((116 * tmp[1]) - 16, 500 * (tmp[0] - tmp[1]), 200 * (tmp[1] - tmp[2]))

    # CIELab -> ScaledCIELab (Ref 3)
    return [
        int(tmp[0] * 65535 / 100 + 0.5),
        int((tmp[1] + 128) * 65535 / 255 + 0.5),
        int((tmp[2] + 128) * 65535 / 255 + 0.5),
    ]
