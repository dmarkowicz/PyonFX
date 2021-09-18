from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Callable, NoReturn, Tuple, TypeVar

import numpy as np

from ..types import NamedMutableSequence, SomeArrayLike

_CT = TypeVar('_CT', bound='Coordinates')


class Coordinates(NamedMutableSequence[float], ABC):
    """Abstract coordinate system"""
    __slots__: Tuple[str, ...] = ()

    def __contains__(self, o: object) -> NoReturn:
        raise NotImplementedError

    @property
    def __self(self) -> Coordinates:
        return self

    def __add__(self: _CT, _p: Tuple[float, ...]) -> _CT:
        return self.__class__(*[sum(r, start=0.0) for r in zip(self.__self, _p)])

    def __radd__(self: _CT, _p: Tuple[float, ...]) -> _CT:
        return self.__add__(_p)

    def __sub__(self: _CT, _p: Tuple[float, ...]) -> _CT:
        return self.__class__(*[c0 - c1 for (c0, c1) in zip(self.__self, _p)])

    def __rsub__(self: _CT, _p: Tuple[float, ...]) -> _CT:
        return self.__class__(*[c1 - c0 for (c0, c1) in zip(self.__self, _p)])

    def __mul__(self: _CT, _p: int | Tuple[float, ...]) -> _CT:
        if isinstance(_p, int):
            nattrs = [getattr(self, attr) * _p for attr in self.__slots__]
        else:
            nattrs = [c0 * c1 for (c0, c1) in zip(self.__self, _p)]
        return self.__class__(*nattrs)

    def __rmul__(self: _CT, _p: int | Tuple[float, ...]) -> _CT:
        return self.__mul__(_p)

    @staticmethod
    def __matmul_func__(_mat1: SomeArrayLike, _mat2: SomeArrayLike) -> map[float]:
        return map(float, np.asarray(_mat1) @ np.asarray(_mat2))

    def __matmul__(self: _CT, _mat: SomeArrayLike) -> _CT:
        return self.__class__(*self.__matmul_func__(self.__self[:len(_mat)], _mat))

    def __rmatmul__(self: _CT, _mat: SomeArrayLike) -> _CT:
        return self.__class__(*self.__matmul_func__(_mat, self.__self[:len(_mat)]))

    def __neg__(self: _CT) -> _CT:
        return self.__class__(*[- a for a in self])

    def __pos__(self: _CT) -> _CT:
        return self.__class__(*[+ a for a in self])

    def __abs__(self: _CT) -> _CT:
        return self.__class__(*[abs(a) for a in self])

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Coordinates):
            return NotImplemented
        return type(self) == type(o) and all(
            getattr(self, ss) == getattr(o, os)
            for ss, os in zip(self.__slots__, o.__slots__)
        )

    def __setattr_with_func__(self, func: Callable[[Coordinates], Any]) -> None:
        for attr, value in zip(self.__slots__, func(self)):
            setattr(self, attr, value)

    def round(self, ndigits: int = 3) -> None:
        """
        Round the Point to a given precision in decimal digits.

        :param ndigits:         Number of digits, defaults to None
        """
        self.__setattr_with_func__(lambda x: np.around(x, ndigits))

    def trunc(self) -> None:
        """
        Truncates the Point to the nearest integer toward 0.
        """
        self.__setattr_with_func__(np.trunc)

    def floor(self) -> None:
        """
        Return the floor of the Point as an integer.

        :return:                New floored Point
        """
        self.__setattr_with_func__(np.floor)

    def ceil(self) -> None:
        """
        Return the ceiling of the Point as an integer.
        """
        self.__setattr_with_func__(np.ceil)

    @abstractmethod
    def to_2d(self) -> Any:
        """
        Convert current coordinate in the 2D cartesian system.
        If the current system has more axis than the output one, the conversion will be lossy,
        meaning that the last axis will just be ignored.
        """
        ...

    @abstractmethod
    def to_3d(self) -> Any:
        """
        Convert current coordinate in the 3D cartesian system.
        If the current system has less axis than the output one, the z axis will be set to 0
        """
        ...

    @abstractmethod
    def to_polar(self) -> Any:
        """
        Convert current coordinate to 2D polar system.
        If the current system has more axis than the output one, the conversion will be lossy,
        meaning that the last axis will just be ignored.
        """
        ...

    @abstractmethod
    def to_cylindrical(self) -> Any:
        """
        Convert current coordinate in the polar cylindrical system.
        If the current system has less axis than the output one, the z axis will be set to 0
        """
        ...

    @abstractmethod
    def to_spherical(self) -> Any:
        """
        Convert current coordinate in the polar spherical system.
        If the current system has less axis than the output one, the theta axis will be set to 90 degrees
        """
        ...


class Axis(IntEnum):
    """Base axis enum"""
    ...
