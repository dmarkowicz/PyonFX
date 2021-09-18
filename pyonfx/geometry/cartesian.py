"""Coordinates cartesian system interface"""

from abc import ABC
from math import cos, radians, sin
from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .coordinates import Axis, Coordinates


class CartesianAxis(Axis):
    """Simple Enum for Cartesian axis"""
    X = 0
    Y = 1
    Z = 2


class Cartesian2D(Coordinates, ABC):
    """Cartesian coordinate system in two dimensions"""

    x: float
    """Abscissa coordinate"""

    y: float
    """Ordinate coordinate"""

    __slots__ = ('x', 'y')

    def __init__(self, _x: float, _y: float) -> None:
        """
        Make a new object in the 2D cartesian system

        :param _x:              Abscissa
        :param _y:              Ordinate
        """
        super().__init__(_x, _y)

    def __rotate__(self, rot: float, axis: CartesianAxis = CartesianAxis.Z, zp: Tuple[float, ...] = (0., 0.)) -> None:
        """
        Rotate on an imaginary Z-axis

        :param rot:             Needed rotation in degrees
        :param zp:              Zero point where the rotation will be performed, defaults to (0, 0)
        """
        theta = radians(rot)
        R = np.array([(cos(theta), -sin(theta)), (sin(theta), cos(theta))], np.float64)
        O, P = np.atleast_2d(zp), np.atleast_2d(self)
        R = cast(NDArray[np.float64], R)
        O, P = cast(NDArray[np.float64], O), cast(NDArray[np.float64], P)
        nvals = map(float, np.squeeze((R @ (P.T - O.T) + O.T).T))  # type: ignore[attr-defined]

        for attr, value in zip(self.__slots__, nvals):
            setattr(self, attr, value)


class Cartesian3D(Cartesian2D, ABC):
    """Cartesian coordinate system in three dimensions"""

    z: float
    """Applicate coordinate"""

    __slots__ = ('x', 'y', 'z')

    def __init__(self, _x: float, _y: float, _z: float) -> None:
        """
        Make a new object in the 3D cartesian system

        :param _x:              Abscissa
        :param _y:              Ordinate
        :param _z:              Applicate
        """
        super().__init__(_x, _y)
        self.z = _z

    def __rotate__(self, rot: float, axis: CartesianAxis = CartesianAxis.Z, zp: Tuple[float, ...] = (0., 0., 0.)) -> None:
        """
        Rotate on given axis

        :param rot:             Needed rotation in degrees
        :param axis:            The rotation will be performed on this axis
        :param zp:              Zero point where the rotation will be performed, defaults to (0, 0, 0)
        """
        rot_mats = (self._rot_mat_x, self._rot_mat_y, self._rot_mat_z)
        try:
            rmat = rot_mats[axis]
        except IndexError as i_err:
            raise ValueError(f'{self.__class__.__name__}: Wrong axis number') from i_err

        R, O, P = rmat(radians(rot)), np.atleast_3d(zp), np.atleast_3d(self)
        O, P = cast(NDArray[np.float64], O), cast(NDArray[np.float64], P)
        nvals = np.squeeze((R @ (P.T - O.T) + O.T).T)  # type: ignore[attr-defined]

        for attr, value in zip(self.__slots__, nvals):
            setattr(self, attr, float(value))

    @staticmethod
    def _rot_mat_x(theta: float) -> NDArray[np.float64]:
        return np.array(
            [(1, 0, 0),
             (0, cos(theta), -sin(theta)),
             (0, sin(theta), cos(theta))], np.float64
        )

    @staticmethod
    def _rot_mat_y(theta: float) -> NDArray[np.float64]:
        return np.array(
            [(cos(theta), 0, sin(theta)),
             (0, 1, 0),
             (-sin(theta), 0, cos(theta))], np.float64
        )

    @staticmethod
    def _rot_mat_z(theta: float) -> NDArray[np.float64]:
        return np.array(
            [(cos(theta), -sin(theta), 0),
             (sin(theta), cos(theta), 0),
             (0, 0, 1)], np.float64
        )