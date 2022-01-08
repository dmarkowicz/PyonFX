"""Miscellaneous and utility functions"""

__all__ = ['clamp_value', 'chunk']

from itertools import islice
from typing import Iterable, Iterator, Literal, Tuple, overload

import numpy as np

from .types import Nb, T_co


def clamp_value(val: Nb, min_val: Nb, max_val: Nb) -> Nb:
    """
    Clamp value val between min_val and max_val

    :param val:         Value to clamp
    :param min_val:     Minimum value
    :param max_val:     Maximum value
    :return:            Clamped value
    """
    # return min(max_val, max(min_val, val))
    return min_val if val < min_val else max_val if val > max_val else val  # type: ignore


@overload
def chunk(iterable: Iterable[T_co], size: Literal[2] = 2) -> Iterator[Tuple[T_co, T_co]]:
    ...


@overload
def chunk(iterable: Iterable[T_co], size: Literal[3]) -> Iterator[Tuple[T_co, T_co, T_co]]:
    ...


def chunk(iterable: Iterable[T_co], size: int = 2) -> Iterator[Tuple[T_co, ...]]:  # type: ignore
    """
    Split an iterable of arbitrary length into equal size chunks

    :param iterable:        Iterable to be splitted it up
    :param size:            Chunk size, defaults to 2
    :return:                Iterator of tuples
    :yield:                 Tuple of size ``size``
    """
    niter = iter(iterable)
    return iter(lambda: tuple(islice(niter, size)), ())


def frange(start: float, end: float, step: float) -> Iterator[float]:
    """
    Floating version of range() built-in

    :param start:           Start value (inclusive)
    :param end:             Stop value (exclusive)
    :param step:            Increment value
    :return:                A iterator of float values
    """
    return iter(map(lambda x: float(round(x, 15)), np.linspace(start, end, round((end-start)/step), endpoint=False)))
