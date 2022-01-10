"""Coordinates polar system interface"""

from abc import ABC
from math import radians

from .coordinates import Axis, Coordinates


class PolarAxis(Axis):
    """Simple Enum for Polar axis"""
    R = 0
    PHI = 1
    Z = 2
    THETA = 2


class Polar(Coordinates, ABC, ignore_slots=True):
    """Polar coordinate system in two dimensions"""

    r: float
    """Axial distance or radial distance"""

    phi: float
    """Azimuth φ angle in radians"""

    def __init__(self, r: float, phi: float) -> None:
        """
        Make a new object in the polar system

        :param _r:              Radial distance
        :param _phi:            Azimuth angle in radians
        """
        super().__init__(r, phi)

    def __rotate__(self, rot: float) -> None:
        """
        Rotate the current object

        :param rot:             Needed rotation in degrees
        """
        self.phi += radians(rot)


class Cylindrical(Polar, ABC, ignore_slots=True):
    """Polar cylindrical coordinate system in three dimensions"""

    z: float
    """Axial coordinate or height"""

    def __init__(self, r: float, phi: float, z: float) -> None:
        """
        Make a new object in the cylindrical system

        :param _r:              Radial distance
        :param _phi:            Azimuth angle in radians
        :param _z:              Height
        """
        super().__init__(r, phi)
        self.z = z

    def __rotate__(self, rot: float) -> None:
        """
        Rotate the current object

        :param rot:             Needed rotation in degrees
        """
        self.phi += radians(rot)


class Spherical(Polar, ABC, ignore_slots=True):
    """Polar spherical coordinate system in three dimensions"""

    theta: float
    """Inclination (or polar angle) θ in radians"""

    def __init__(self, r: float, phi: float, theta: float) -> None:
        """
        Make a new object in the spherical system

        :param r:                Radial distance
        :param phi:              Azimuth angle in radians
        :param theta:            Inclinaison in radians
        """
        super().__init__(r, phi)
        self.theta = theta

    def __rotate__(self, rot: float, axis: PolarAxis = PolarAxis.PHI) -> None:
        """
        Rotate the current object

        :param rot:             Needed rotation in degrees
        """
        if axis == PolarAxis.PHI:
            self.phi += radians(rot)
        elif axis == PolarAxis.THETA:
            self.theta += radians(rot)
        else:
            raise ValueError(f'{self.__class__.__name__}: Axis not supported')
