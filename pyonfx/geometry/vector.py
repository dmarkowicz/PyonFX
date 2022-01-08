"""Vector interface"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import atan2, cos, radians, sin, sqrt
from typing import TypeVar

import numpy as np

from .cartesian import Cartesian2D, Cartesian3D
from .coordinates import Coordinates
from .polar import Cylindrical, Polar, Spherical


VectorT = TypeVar('VectorT', bound='Vector')


class Vector(Coordinates, ABC):
    """Abstract Point Vector with"""

    @property
    @abstractmethod
    def norm(self) -> float:
        """
        Get the vector length (or norm)

        :return:            Vector length
        """
        ...

    @abstractmethod
    def __angle__(self: VectorT, v: VectorT) -> float:
        """
        Get angle vetween the current vector and another vector
        with sign by clockwise direction

        :param v:           Another vector
        :return:            Angle in radians
        """
        ...

    @abstractmethod
    def __orthogonal__(self: VectorT, v: VectorT) -> float | VectorT:
        """
        Get orthogonal vector of two vectors

        :param v:           Another vector

        :return:            Scalar if 2D dimensions else a vector
        """
        ...

    def __stretch__(self: VectorT, length: float) -> VectorT:
        """
        Scale vector to given length

        :param length:      Required length
        :return:            Vector stretched
        """
        if (norm := self.norm) == 0.:
            return self.__class__(*[0] * self.__len__())
        return self.__class__(*[c * length / norm for c in self])


class VectorCartesian2D(Vector, Cartesian2D):
    """Vector in Cartesian2D system"""

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self))

    def __angle__(self, v: VectorCartesian2D) -> float:
        # https://stackoverflow.com/a/35134034
        angle = atan2(np.linalg.det(np.asarray((self, v))), np.dot(self, v))
        # Return with sign by clockwise direction
        return - angle if np.cross(self, v) < 0 else angle

    def __orthogonal__(self, v: VectorCartesian2D) -> float:
        return float(np.cross(self, v))

    def to_2d(self) -> VectorCartesian2D:
        return self

    def to_3d(self) -> VectorCartesian3D:
        return VectorCartesian3D(self.x, self.y, 0.)

    def to_polar(self) -> VectorPolar:
        return VectorPolar(
            sqrt(self.x ** 2 + self.y ** 2),
            atan2(self.y, self.x)
        )

    def to_cylindrical(self) -> VectorCylindrical:
        return self.to_polar().to_cylindrical()

    def to_spherical(self) -> VectorSpherical:
        return self.to_polar().to_spherical()


class VectorCartesian3D(Vector, Cartesian3D):
    """Vector in Cartesian3D system"""

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self))

    def __angle__(self, v: VectorCartesian3D) -> float:
        norm0, norm1 = self.norm, v.norm
        if norm0 * norm1 == 0:
            return 0.
        angle = float(np.arccos(np.clip(np.dot(self, v) / norm0 / norm1, -1., 1.)))
        # Return with sign by clockwise direction
        return - angle if np.cross(self, v)[-1] < 0 else angle  # type: ignore[index]

    def __orthogonal__(self, v: VectorCartesian3D) -> VectorCartesian3D:
        return VectorCartesian3D(*map(float, np.cross(self, v)))

    def to_2d(self) -> VectorCartesian2D:
        return VectorCartesian2D(self.x, self.y)

    def to_3d(self) -> VectorCartesian3D:
        return self

    def to_polar(self) -> VectorPolar:
        return self.to_2d().to_polar()

    def to_cylindrical(self) -> VectorCylindrical:
        return VectorCylindrical(
            sqrt(self.x ** 2 + self.y ** 2),
            atan2(self.y, self.x),
            self.z
        )

    def to_spherical(self) -> VectorSpherical:
        x2y2 = self.x ** 2 + self.y ** 2
        return VectorSpherical(
            sqrt(x2y2 + self.z ** 2),
            atan2(self.y, self.x),
            atan2(x2y2, self.z)
        )


class VectorPolar(Vector, Polar):
    """Vector in Polar system"""

    @property
    def norm(self) -> float:
        return self.r

    def __angle__(self, v: VectorPolar) -> float:
        return self.to_2d().__angle__(v.to_2d())

    def __orthogonal__(self, v: VectorPolar) -> float:
        return self.to_2d().__orthogonal__(v.to_2d())

    def __stretch__(self, length: float) -> VectorPolar:
        return VectorPolar(length, self.phi)

    def to_2d(self) -> VectorCartesian2D:
        return VectorCartesian2D(
            self.r * cos(self.phi),
            self.r * sin(self.phi)
        )

    def to_3d(self) -> VectorCartesian3D:
        return self.to_2d().to_3d()

    def to_polar(self) -> VectorPolar:
        return self

    def to_cylindrical(self) -> VectorCylindrical:
        return VectorCylindrical(self.r, self.phi, 0)

    def to_spherical(self) -> VectorSpherical:
        return VectorSpherical(
            self.r,
            self.phi,
            radians(90.)
        )


class VectorCylindrical(Vector, Cylindrical):
    """Vector in Cylindrical system"""

    @property
    def norm(self) -> float:
        return self.r

    def __angle__(self, v: VectorCylindrical) -> float:
        return atan2(v.z, v.r) - atan2(self.z, self.r)

    def __orthogonal__(self, v: VectorCylindrical) -> VectorCylindrical:
        return self.to_3d().__orthogonal__(v.to_3d()).to_cylindrical()

    def __stretch__(self, length: float) -> VectorCylindrical:
        return self.to_3d().__stretch__(length).to_cylindrical()

    def to_2d(self) -> VectorCartesian2D:
        return VectorCartesian2D(
            self.r * cos(self.phi),
            self.r * sin(self.phi)
        )

    def to_3d(self) -> VectorCartesian3D:
        return VectorCartesian3D(
            self.r * cos(self.phi),
            self.r * sin(self.phi),
            self.z
        )

    def to_polar(self) -> VectorPolar:
        return VectorPolar(self.r, self.phi)

    def to_cylindrical(self) -> VectorCylindrical:
        return self

    def to_spherical(self) -> VectorSpherical:
        return VectorSpherical(
            sqrt(self.r ** 2 + self.z ** 2),
            self.phi,
            atan2(self.r, self.z)
        )


class VectorSpherical(Vector, Spherical):
    """Vector in Spherical system"""

    @property
    def norm(self) -> float:
        return self.r * sin(self.theta)

    def __angle__(self, v: VectorSpherical) -> float:
        return v.theta - self.theta

    def __orthogonal__(self, v: VectorSpherical) -> VectorSpherical:
        return self.to_3d().__orthogonal__(v.to_3d()).to_spherical()

    def __stretch__(self, length: float) -> VectorSpherical:
        return self.to_3d().__stretch__(length).to_spherical()

    def to_2d(self) -> VectorCartesian2D:
        return self.to_3d().to_2d()

    def to_3d(self) -> VectorCartesian3D:
        return VectorCartesian3D(
            self.r * sin(self.theta) * cos(self.phi),
            self.r * sin(self.theta) * sin(self.phi),
            self.r * cos(self.theta)
        )

    def to_polar(self) -> VectorPolar:
        return VectorPolar(
            self.r * sin(self.theta),
            self.phi,
        )

    def to_cylindrical(self) -> VectorCylindrical:
        return VectorCylindrical(
            self.r * sin(self.theta),
            self.phi,
            self.r * cos(self.theta)
        )

    def to_spherical(self) -> VectorSpherical:
        return self
