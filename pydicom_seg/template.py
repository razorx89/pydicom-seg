from typing import Iterable, Union

import json
import os

import jsonschema
import pydicom


def _create_validator():
    # Load both common and segmentation schema files
    schemas_dir = os.path.join(
        os.path.dirname(__file__),
        'externals',
        'dcmqi',
        'doc',
        'schemas'
    )
    seg_schema_path = os.path.join(schemas_dir, 'seg-schema.json')
    with open(seg_schema_path) as ifile:
        seg_schema = json.load(ifile)
    with open(os.path.join(schemas_dir, 'common-schema.json')) as ifile:
        common_schema = json.load(ifile)

    # Create validator with pre-loaded schema store
    return jsonschema.Draft4Validator(
        seg_schema,
        resolver=jsonschema.RefResolver(
            base_uri='file://' + seg_schema_path,
            referrer=seg_schema,
            store={
                seg_schema['id']: seg_schema,
                common_schema['id']: common_schema
            }
        )
    )


def _create_code_sequence(json: dict) -> pydicom.Sequence:
    dataset = pydicom.Dataset()
    for key in json:
        dataset.__setattr__(key, json[key])
    return pydicom.Sequence([dataset])


def _create_segment_dataset(json: dict) -> pydicom.Dataset:
    dataset = pydicom.Dataset()

    # Mandatory fields
    dataset.SegmentAlgorithmName = json.get('SegmentAlgorithmName', '')
    dataset.SegmentAlgorithmType = json.get('SegmentAlgorithmType', 'SEMIAUTOMATIC')
    dataset.SegmentNumber = json['labelID']
    dataset.SegmentedPropertyCategoryCodeSequence = _create_code_sequence(
        json['SegmentedPropertyCategoryCodeSequence']
    )
    dataset.SegmentedPropertyTypeCodeSequence = _create_code_sequence(
        json['SegmentedPropertyTypeCodeSequence']
    )

    # Optional fields
    optional_code_sequences = [
        'SegmentedPropertyTypeModifierCodeSequence',
        'AnatomicRegionSequence',
        'AnatomicRegionModifierSequence'
    ]
    for code_sequence in optional_code_sequences:
        if code_sequence in json:
            dataset.__setattr__(code_sequence, json[code_sequence])

    optional_tags_no_default = [
        'SegmentDescription',
        'SegmentLabel',
        'TrackingIdentifier',
        'TrackingUniqueIdentifier'
    ]
    for tag_name in optional_tags_no_default:
        if tag_name in json:
            dataset.__setattr__(tag_name, json[tag_name])

    if 'RecommendedDisplayCIELabValue' in json:
        dataset.RecommendedDisplayCIELabValue = json['RecommendedDisplayCIELabValue']
    elif 'recommendedDisplayRGBValue' in json:
        dataset.RecommendedDisplayCIELabValue = rgb_to_cielab(
            json['recommendedDisplayRGBValue']
        )

    return dataset


def from_dcmqi_metainfo(metainfo: Union[dict, str]) -> pydicom.Dataset:
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
        ('BodyPartExamined', ''),
        ('ClinicalTrialCoordinatingCenterName', ''),
        ('ClinicalTrialSeriesID', 'Session1'),
        ('ClinicalTrialTimePointID', '1'),
        ('ContentCreatorName', 'Reader1'),
        ('ContentDescription', 'Image segmentation'),
        ('ContentLabel', 'SEGMENTATION'),
        ('InstanceNumber', '1'),
        ('SeriesDescription', 'Segmentation'),
        ('SeriesNumber', '300')
    ]

    for tag_name, default_value in tags_with_defaults:
        if default_value is None:
            if tag_name in metainfo:
                dataset.__setattr__(tag_name, metainfo[tag_name])
        else:
            dataset.__setattr__(tag_name, metainfo.get(tag_name, default_value))

    dataset.SegmentSequence = pydicom.Sequence([
        _create_segment_dataset(x[0]) for x in metainfo['segmentAttributes']
    ])

    return dataset


def rgb_to_cielab(rgb: Iterable[int]):
    """
    Convert from RGB to CIELab color space.

    Arguments:
        rgb: Iterable of length 3 with integer values between 0 and 255

    References:
        - https://github.com/QIICR/dcmqi/blob/0c101b702f12a86cc142cb000a074fbd341f8784/libsrc/Helper.cpp#L173
        - https://github.com/QIICR/dcmqi/blob/0c101b702f12a86cc142cb000a074fbd341f8784/libsrc/Helper.cpp#L256
        - https://github.com/QIICR/dcmqi/blob/0c101b702f12a86cc142cb000a074fbd341f8784/libsrc/Helper.cpp#L336
    """
    assert len(rgb) == 3

    # RGB -> CIEXYZ (Ref 1)
    tmp = tuple(min(max(x / 255.0, 0.0), 1.0) for x in rgb)
    tmp = tuple(((x + 0.055) / 1.055)**2.4 if x > 0.04045 else x / 12.92 for x in tmp)
    tmp = tuple(x * 100 for x in tmp)
    tmp = (
        0.4124564 * tmp[0] + 0.3575761 * tmp[1] + 0.1804375 * tmp[2],
        0.2126729 * tmp[0] + 0.7151522 * tmp[1] + 0.0721750 * tmp[2],
        0.0193339 * tmp[0] + 0.1191920 * tmp[1] + 0.9503041 * tmp[2]
    )

    # CIEXYZ -> CIELab (Ref 2)
    tmp = tuple(x / y for x, y in zip(tmp, (95.047, 100, 108.883)))
    tmp = tuple(x**(1/3) if x > 0.008856 else (7.787*x) + (16 / 116) for x in tmp)
    tmp = (
        (116 * tmp[1]) - 16,
        500 * (tmp[0] - tmp[1]),
        200 * (tmp[1] - tmp[2])
    )

    # CIELab -> ScaledCIELab (Ref 3)
    return (
        int(tmp[0] * 65535 / 100 + 0.5),
        int((tmp[1] + 128) * 65535 / 255 + 0.5),
        int((tmp[2] + 128) * 65535 / 255 + 0.5)
    )
