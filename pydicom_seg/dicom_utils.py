from typing import List, Optional, Union

import numpy as np
import pydicom
import SimpleITK as sitk


class CodeSequence(pydicom.Sequence):
    """Helper class for constructing a DICOM CodeSequence."""

    def __init__(self, value: str, scheme_designator: str, meaning: str):
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
    def add_dimension(
        self,
        dimension_index_pointer: Union[str, pydicom.tag.Tag],
        functional_group_pointer: Optional[Union[str, pydicom.tag.Tag]] = None,
    ) -> None:
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
        ds.DimensionDescriptionLabel = (
            pydicom.datadict.keyword_for_tag(dimension_index_pointer)
            or f"Unknown tag {dimension_index_pointer}"
        )

        if functional_group_pointer is not None:
            if isinstance(functional_group_pointer, str):
                functional_group_pointer = pydicom.tag.Tag(
                    pydicom.datadict.tag_for_keyword(functional_group_pointer)
                )
            ds.FunctionalGroupPointer = functional_group_pointer

        self.append(ds)


def dcm_to_sitk_orientation(iop: List[str]) -> np.ndarray:
    assert len(iop) == 6

    # Extract x-vector and y-vector
    x_dir = [float(x) for x in iop[:3]]
    y_dir = [float(x) for x in iop[3:]]

    # L2 normalize x-vector and y-vector
    x_dir /= np.linalg.norm(x_dir)
    y_dir /= np.linalg.norm(y_dir)

    # Compute perpendicular z-vector
    z_dir = np.cross(x_dir, y_dir)

    return np.stack([x_dir, y_dir, z_dir], axis=1)


def sitk_to_dcm_orientation(img: sitk.Image) -> List[str]:
    direction = img.GetDirection()
    assert len(direction) == 9
    direction = np.asarray(direction).reshape((3, 3))
    orientation = direction.T[:2]
    return [f"{x:e}" for x in orientation.ravel()]
