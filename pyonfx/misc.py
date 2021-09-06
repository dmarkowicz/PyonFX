"""Miscellaneous and utility functions"""

__all__ = ['clamp_value', 'chunk']

from itertools import islice
from typing import Iterable, Iterator, Literal, Tuple, overload

from .types import Nb, T_co


def clamp_value(val: Nb, min_val: Nb, max_val: Nb) -> Nb:
    """
    Clamp value val between min_val and max_val

    :param val:         Value to clamp
    :param min_val:     Minimum value
    :param max_val:     Maximum value
    :return:            Clamped value
    """
    return min(max_val, max(min_val, val))


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
