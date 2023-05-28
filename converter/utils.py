import math
from itertools import chain
from typing import Tuple, List, Any, Iterable


def calculate_dict_hash(dictionary: dict) -> int:
    return hash(frozenset(dictionary.items()))


def euler_rotation_to_quaternion(
    euler_rotation: Tuple[float, float, float]
) -> Tuple[float, float, float, float]:
    if euler_rotation is None:
        return None
    x, y, z = euler_rotation

    c1 = math.cos(x / 2)
    c2 = math.cos(y / 2)
    c3 = math.cos(z / 2)
    s1 = math.sin(x / 2)
    s2 = math.sin(y / 2)
    s3 = math.sin(z / 2)

    return (
        s1 * c2 * c3 + c1 * s2 * s3,
        c1 * s2 * c3 - s1 * c2 * s3,
        c1 * c2 * s3 + s1 * s2 * c3,
        c1 * c2 * c3 - s1 * s2 * s3,
    )


def flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    return list(chain.from_iterable(list_of_lists))


def get_image_mimetype(raw_image: bytes) -> str:
    if raw_image.startswith(b"\x89PNG"):
        return "image/png"
    elif raw_image.startswith(b"\xff\xd8"):
        return "image/jpeg"
    else:
        raise ValueError(f"Couldn't recognise image type")


def next_filter(iter: Iterable[Any], func: callable) -> Any:
    try:
        return next(filter(func, iter))
    except StopIteration:
        return None
