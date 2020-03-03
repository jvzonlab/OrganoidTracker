from typing import Union, Tuple

MPLColor = Union[
    Tuple[float, float, float],
    Tuple[float, float, float, float],
    str,
    float
]

# Primitive types that can be stored directly in JSON files.
DataType = Union[
    float,
    int,
    str,
    bool
]
