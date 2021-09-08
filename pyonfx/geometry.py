"""Geometry module"""
from __future__ import annotations

__all__ = [
    'curve4_to_lines', 'split_line',
    'rotate_point', 'get_vector_angle', 'get_vector_length',
    'make_ellipse', 'make_parallelogram', 'make_triangle'
]

from functools import reduce
from itertools import chain
from math import asin, atan2, ceil, cos, degrees, dist, inf, radians, sin, sqrt
from typing import List, Tuple, cast, overload

import cv2
import numpy as np

from .misc import chunk
from .types import AssBCurve, BézierCoord


def curve4_to_lines(b_coord: BézierCoord, tolerance: float, /) -> List[Tuple[float, float]]:
    """function to convert 4th degree curve to line points"""

    ncoord: List[Tuple[float, float]] = []

    def _curve4_subdivide(b_coord: BézierCoord, /) -> Tuple[BézierCoord, BézierCoord]:
        """4th degree curve subdivider (De Casteljau)"""
        # Calculate points on curve vectors
        lcoord = list(chain.from_iterable(b_coord))
        sub3 = [sum(c) / 2 for c in zip(lcoord, lcoord[2:])]
        sub2 = [sum(c) / 2 for c in zip(sub3, sub3[2:])]
        subx1, suby1 = [sum(c) / 2 for c in zip(sub2, sub2[2:])]

        # Return new 2 curves
        b0 = b_coord[0], (sub3[0], sub3[1]), (sub2[0], sub2[1]), (subx1, suby1)
        b1 = (subx1, suby1), (sub2[-2], sub2[-1]), (sub3[-2], sub3[-1]), b_coord[-1]
        return b0, b1

    def _curve4_is_flat(b_coord: BézierCoord, /) -> bool:
        """Check flatness of 4th degree curve with angles"""
        lcoord = list(chain.from_iterable(b_coord))
        # Pack curve vectors (only ones non zero)
        vecs = [
            reduce(lambda a, b: b - a, coord)
            for coord in zip(lcoord, lcoord[2:])
        ]
        vecsp = [
            (v0, v1) for v0, v1 in chunk(vecs, 2)
            if not (v0 == 0 and v1 == 0)
        ]
        # # Old code:
        # (x0, y0), (x1, y1), (x2, y2), (x3, y3) = b_coord
        # vecs = [[x1 - x0, y1 - y0], [x2 - x1, y2 - y1], [x3 - x2, y3 - y2]]
        # vecsp = [el for el in vecs if not (el[0] == 0 and el[1] == 0)]

        # Check flatness on vectors
        vecsp.reverse()
        for v0, v1 in reversed(list(zip(vecsp[1:], vecsp))):
            if abs(get_vector_angle(v0, v1)) > tolerance:
                return False
        return True

    def _convert_recursive(b_coord: BézierCoord, /) -> None:
        """Conversion in recursive processing"""
        if _curve4_is_flat(b_coord):
            ncoord.append(b_coord[-1])
            return None
        b0, b1 = _curve4_subdivide(b_coord)
        _convert_recursive(b0)
        _convert_recursive(b1)

    # Splitting curve recursively until we're not satisfied (angle <= tolerance)
    _convert_recursive(b_coord)
    return ncoord


def split_line(p0: Tuple[float, float], p1: Tuple[float, float], max_length: float) -> List[Tuple[float, float]]:
    """Split a line (p0, p1) into shorter lines with maximum max_length"""
    ncoord: List[Tuple[float, float]] = []

    distance = dist(p0, p1)
    if distance > max_length:
        (x0, y0), (x1, y1) = p0, p1
        # Equal step between the two points instead of having all points to 16
        # except the last for the remaining distance
        step = distance / ceil(distance / max_length)
        # Step can be a float so numpy.arange is prefered
        for i in np.arange(step, distance, step):
            pct = i / distance
            ncoord.append(((x0 + (x1 - x0) * pct), (y0 + (y1 - y0) * pct)))
    ncoord.append(p1)
    return ncoord


def rotate_and_project(x: float, y: float, z: float, rotations: Tuple[float, float, float],
                       origin: Tuple[float, float] = (0., 0.),
                       offset: Tuple[float, float] = (0., 0.)) -> Tuple[float, float]:
    """
    Rotate point by given rotations on all axis

    :param x:               Abscissa of the point
    :param y:               Ordinate of the point
    :param z:               Applicate of the point
    :param rotations:       Rotations in degrees in this order of X, Y and Z axis
    :param origin:          Origin anchor, defaults to (0., 0.)
    :param offset:          Offset coordinates, defaults to (0., 0.)
    :return:                A tuple of 2D rotated coordinates
    """
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints
    # Length of the camera seems to be 310 according to my tests
    img_pts, _ = cv2.projectPoints(
        objectPoints=np.array((x, y, z), np.float32),
        rvec=np.array([radians(r) for r in rotations], np.float32),
        tvec=np.array((*[o * -1 for o in origin], -310), np.float32),
        cameraMatrix=np.array([(-310, 0, offset[0]), (0, -310, offset[1]), (0, 0, 1)], np.float32),
        distCoeffs=np.empty((4, 1), np.float32),
    )
    return img_pts[0][0][0], img_pts[0][0][1]


def rotate_point(x: float, y: float, zpx: float, zpy: float, rotation: float) -> Tuple[float, float]:
    """
    Rotate a single point on the Z-axis

    :param x:               Abscissa of the point
    :param y:               Ordinate of the point
    :param zpx:             Abscissa of the zero point
    :param zpy:             Ordinate of the zero point
    :param rotation:        Rotation in degrees
    :return:                A tuple of rotated coordinates
    """
    # Distance to zero-point
    zpd = dist((zpx, zpy), (x, y))

    rot = radians(rotation)
    curot = atan2(y - zpy, x - zpx)

    nx = zpd * cos(curot + rot) + zpx
    ny = zpd * sin(curot + rot) + zpy
    return nx, ny


def get_vector_angle(v0: Tuple[float, float], v1: Tuple[float, float]) -> float:
    """Get angle between two vectors"""
    # https://stackoverflow.com/a/35134034

    angler = atan2(np.linalg.det((v0, v1)), np.dot(v0, v1))
    angled = degrees(angler)

    # Return with sign by clockwise direction
    return - angled if np.cross(v0, v1) < 0 else angled


def get_vector_length(vector: Tuple[float, float] | Tuple[float, float, float]) -> float:
    """
    Get a vector length (or norm)

    :param vector:      Vector
    :return:            Vector length
    """
    return cast(float, np.linalg.norm(vector))


def get_vector(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float]:
    """
    Make a vector fom two points

    :param p0:          First point
    :param p1:          Second point
    :return:            A vector
    """
    return float(p0[0] - p1[0]), float(p0[1] - p1[1])


@overload
def get_ortho_vector(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float]:
    """
    Get orthogonal vector of two points

    :param p0:          First point
    :param p1:          Second point
    :return:            Orthogonal vector coordinates
    """
    ...


@overload
def get_ortho_vector(p0: Tuple[float, float, float], p1: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Get orthogonal vector of two points

    :param p0:          First point
    :param p1:          Second point
    :return:            Orthogonal vector coordinates
    """
    ...


def get_ortho_vector(p0: Tuple[float, float] | Tuple[float, float, float],
                     p1: Tuple[float, float] | Tuple[float, float, float]
                     ) -> Tuple[float, float] | Tuple[float, float, float]:
    cross = np.cross(p0, p1)
    if len(cross) == 3:
        x, y, z = cross
        return x, y, z
    x, y = cross
    return x, y


@overload
def stretch_vector(v: Tuple[float, float], length: float) -> Tuple[float, float]:
    """
    Scale vector to given length

    :param v:           Vector
    :param length:      Required length
    :return:            Vector stretched
    """
    ...


@overload
def stretch_vector(v: Tuple[float, float, float], length: float) -> Tuple[float, float, float]:
    """
    Scale vector to given length

    :param v:           Vector
    :param length:      Required length
    :return:            Vector stretched
    """
    ...


def stretch_vector(v: Tuple[float, float] | Tuple[float, float, float],
                   length: float) -> Tuple[float, float] | Tuple[float, float, float]:
    cur_length = get_vector_length(v)
    if cur_length == 0.:
        return (0., 0.) if len(v) == 2 else (0., 0., 0.)
    factor = length / cur_length
    return tuple(c * factor for c in v)  # type: ignore


def get_line_intersect(p0: Tuple[float, float], p1: Tuple[float, float],
                       p2: Tuple[float, float], p3: Tuple[float, float],
                       strict: bool = True) -> Tuple[float, float]:
    """
    Get line intersection coordinates between 4 points

    :param p0:          First point
    :param p1:          Second point
    :param p2:          Third point
    :param p3:          Fourth point
    :param strict:      ???, defaults to True
    :return:            Coordinates of the intersection
    """
    v0, v1 = get_vector(p0, p1), get_vector(p2, p3)
    if v0 == (0., 0.) or v1 == (0., 0.):
        raise ValueError('lines mustn\'t have zero length')
    det = np.linalg.det((v0, v1))
    if det != 0:
        pre, post = np.linalg.det((p0, p1)), np.linalg.det((p2, p3))
        ix, iy = (pre * v1[0] - v0[0] * post) / det, (pre * v1[1] - v0[1] * post) / det
        if strict:
            s = (ix - p1[0]) / v0[0] if v0[0] != 0 else (iy - p1[1]) / v0[1]
            t = (ix - p3[0]) / v1[0] if v1[0] != 0 else (iy - p3[1]) / v1[1]
            if s < 0 or s > 1 or t < 0 or t > 1:
                return inf, inf
    else:
        return inf, inf
    return ix, iy


def make_ellipse(
    w: float, h: float, c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True
) -> Tuple[Tuple[float, float], AssBCurve, AssBCurve, AssBCurve, AssBCurve]:
    """
    Make ellipse coordinates with given width and height, centered around (c_xy)

    :param w:               Width of the ellipse
    :param h:               Height of the ellipse
    :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
    :param clockwise:       Direction of point creation, defaults to True
    :return:                Ellipses coordinates
    """
    c = 0.551915024494  # https://spencermortensen.com/articles/bezier-circle/
    cx, cy = c_xy
    cl = - int((-1) ** clockwise)
    return (
        ((cx - 0) * cl, cy + h),
        (
            ((cx - w * c) * cl, cy + h),
            ((cx - w) * cl, cy + h * c),
            ((cx - w) * cl, cy - 0)
        ),
        (
            ((cx - w) * cl, cy - h * c),
            ((cx - w * c) * cl, cy - h),
            ((cx - 0) * cl, cy - h)
        ),
        (
            ((cx + w * c) * cl, cy - h),
            ((cx + w) * cl, cy - h * c),
            ((cx + w) * cl, cy - 0)
        ),
        (
            ((cx + w) * cl, cy + h * c),
            ((cx + w * c) * cl, cy + h),
            ((cx - 0) * cl, cy + h)
        )
    )


def make_parallelogram(
    w: float, h: float, angle: float, c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Make parallelogram coordinates with given width, height and angle, centered around (c_xy)

    :param w:               Width of the parallelogram
    :param h:               Height of the parallelogram
    :param angle:           First angle of the parallelogram in degrees
    :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
    :param clockwise:       Direction of point creation, defaults to True
    :return:                A Shape object representing a parallelogram
    """
    cl = - int((-1) ** clockwise)
    cx, cy = c_xy

    l = h / cos(radians(90 - angle))
    x0, y0 = 0, 0
    x1, y1 = l * cos(radians(angle)), l * sin(radians(angle))
    x2, y2 = x1 + w, y1
    x3, y3 = w, 0

    return (
        ((x0 + cx) * cl, y0 + cy),
        ((x1 + cx) * cl, y1 + cy),
        ((x2 + cx) * cl, y2 + cy),
        ((x3 + cx) * cl, y3 + cy),
        ((x0 + cx) * cl, y0 + cy)
    )


def make_triangle(
    side: float | Tuple[float, float], angle: Tuple[float, float] | float,
    c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Make general triangle coordinates with given sides and angles, centered around (c_xy)

    :param side:            Side(s) of the triangle
    :param angle:           Angle(s) of the triangle in degrees
    :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
    :param clockwise:       Direction of point creation, defaults to True
    :param orthocentred:    Centred in the orthocenter, defaults to True
    :return:                A Shape object representing a triangle
    """
    cl = - int((-1) ** clockwise)
    cx, cy = c_xy

    if isinstance(side, (int, float)) and isinstance(angle, tuple):
        A, B = angle
        C = 180 - A - B
        ab = side
        bc = sin(radians(A)) * ab / sin(radians(C))

        x0, y0 = 0, 0
        x1, y1 = ab, 0
        x2, y2 = x1 - bc * cos(radians(B)), bc * sin(radians(B))
    elif isinstance(side, tuple) and isinstance(angle, (int, float)):
        ab, ac = side
        A = angle
        bc = sqrt(ac ** 2 + ab ** 2 - 2 * ac * ab * cos(radians(A)))
        Br = asin(sin(radians(A)) * ac / bc)

        x0, y0 = 0, 0
        x1, y1 = ab, 0
        x2, y2 = x1 - bc * cos(Br), bc * sin(Br)
    else:
        raise ValueError('make_triangle: possibles values are one side and two angles or two sides and one angle')

    return (
        ((x0 + cx) * cl, y0 + cy),
        ((x1 + cx) * cl, y1 + cy),
        ((x2 + cx) * cl, y2 + cy),
        ((x0 + cx) * cl, y0 + cy)
    )
