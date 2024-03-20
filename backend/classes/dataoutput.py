"""
    DataOutput class
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203,C0412

from dataclasses import dataclass

@dataclass
class DataOutput:
    """
        DataOutput class
    """
    output : list[str, str]
    tokens : int
