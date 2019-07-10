from typing import Union, Tuple

MPLColor = Union[
    Tuple[float, float, float],
    Tuple[float, float, float, float],
    str,
    float
]

# Types that are used as metadata for positions. They can safely be stored.
DataType = Union[
    float,
    int,
    str,
    bool
]
