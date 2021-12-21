__version__ = "0.4.0"

from pydicom_seg import template
from pydicom_seg.reader import MultiClassReader, SegmentReader
from pydicom_seg.writer import MultiClassWriter

__all__ = [
    "MultiClassReader",
    "MultiClassWriter",
    "SegmentReader",
    "template",
]
