import json
import pathlib

from pydicom_seg.template import from_dcmqi_metainfo


class TestTemplate:
    def test_segment_dataset_optional_code_sequences(self) -> None:
        path = (
            pathlib.Path(__file__).parent.parent
            / "pydicom_seg"
            / "externals"
            / "dcmqi"
            / "doc"
            / "examples"
            / "seg-example_multiple_segments_single_input_file.json"
        )
        with path.open() as ifile:
            json_data = json.load(ifile)

        json_data["segmentAttributes"][0][0].update(
            {
                "AnatomicRegionSequence": {
                    "CodeValue": "10200004",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "Liver",
                },
                "AnatomicRegionModifierSequence": {
                    "CodeValue": "7771000",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "Left",
                },
                "SegmentedPropertyTypeModifierCodeSequence": {
                    "CodeValue": "1234567",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "DUMMY",
                },
            }
        )

        meta = from_dcmqi_metainfo(json_data)

        assert "AnatomicRegionSequence" in meta.SegmentSequence[0]
        seq = meta.SegmentSequence[0].AnatomicRegionSequence
        assert len(seq) == 1
        assert seq[0].CodeValue == "10200004"
        assert seq[0].CodingSchemeDesignator == "SCT"
        assert seq[0].CodeMeaning == "Liver"

        assert "AnatomicRegionModifierSequence" in meta.SegmentSequence[0]
        seq = meta.SegmentSequence[0].AnatomicRegionModifierSequence
        assert len(seq) == 1
        assert seq[0].CodeValue == "7771000"
        assert seq[0].CodingSchemeDesignator == "SCT"
        assert seq[0].CodeMeaning == "Left"

        assert "SegmentedPropertyTypeModifierCodeSequence" in meta.SegmentSequence[0]
        seq = meta.SegmentSequence[0].SegmentedPropertyTypeModifierCodeSequence
        assert len(seq) == 1
        assert seq[0].CodeValue == "1234567"
        assert seq[0].CodingSchemeDesignator == "SCT"
        assert seq[0].CodeMeaning == "DUMMY"
