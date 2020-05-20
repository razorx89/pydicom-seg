from typing import Optional, Union

import pydicom


class CodeSequence(pydicom.Sequence):
    """Helper class for constructing a DICOM CodeSequence."""
    def __init__(self,
                 value: str,
                 scheme_designator: str,
                 meaning: str):
        """Creates a code sequence from mandatory arguments.

        Args:
            value: (0x0008, 0x0100) CodeValue
            scheme_designator: (0x0008, 0x0102) CodingSchemeDesignator
            meaning: (0x0008, 0x0104) CodeMeaning
        """
        super().__init__()
        ds = pydicom.Dataset()
        ds.CodeValue = value
        ds.CodingSchemeDesignator = scheme_designator
        ds.CodeMeaning = meaning
        self.append(ds)


class DimensionOrganizationSequence(pydicom.Sequence):
    def add_dimension(self,
                      dimension_index_pointer: Union[str, pydicom.tag.Tag],
                      functional_group_pointer: Optional[Union[str, pydicom.tag.Tag]] = None) -> None:
        ds = pydicom.Dataset()
        if len(self) > 0:
            ds.DimensionOrganizationUID = self[0].DimensionOrganizationUID
        else:
            ds.DimensionOrganizationUID = pydicom.uid.generate_uid()

        if isinstance(dimension_index_pointer, str):
            dimension_index_pointer = pydicom.tag.Tag(
                pydicom.datadict.tag_for_keyword(dimension_index_pointer)
            )
        ds.DimensionIndexPointer = dimension_index_pointer
        ds.DimensionDescriptionLabel = pydicom.datadict.keyword_for_tag(
            dimension_index_pointer
        ) or f'Unknown tag {dimension_index_pointer}'

        if functional_group_pointer is not None:
            if isinstance(functional_group_pointer, str):
                functional_group_pointer = pydicom.tag.Tag(
                    pydicom.datadict.tag_for_keyword(functional_group_pointer)
                )
            ds.FunctionalGroupPointer = functional_group_pointer

        self.append(ds)
