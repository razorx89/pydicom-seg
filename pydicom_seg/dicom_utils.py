from typing import Optional, Union

import pydicom


class CodeSequence(pydicom.Sequence):
    def __init__(self,
                 value: str,
                 scheme_designator: str,
                 meaning: str):
        super().__init__()
        ds = pydicom.Dataset()
        ds.CodeValue = value
        ds.CodingSchemeDesignator = scheme_designator
        ds.CodeMeaning = meaning
        self.append(ds)


class DimensionOrganizationSequence(pydicom.Sequence):
    def add_dimension(self,
                      dimension_index_pointer: Union[str, pydicom.tag.Tag],
                      functional_group_pointer: Optional[Union[str, pydicom.tag.Tag]] = None):
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

        if functional_group_pointer is not None:
            if isinstance(functional_group_pointer, str):
                functional_group_pointer = pydicom.tag.Tag(
                    pydicom.datadict.tag_for_keyword(functional_group_pointer)
                )
            ds.FunctionalGroupPointer = functional_group_pointer

        self.append(ds)
