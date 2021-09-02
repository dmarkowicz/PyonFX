# PyonFX: An easy way to create KFX (Karaoke Effects) and complex typesetting using the ASS format (Advanced Substation Alpha).
# Copyright (C) 2019 Antonio Strippoli (CoffeeStraw/YellowFlash)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyonFX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from __future__ import annotations

__all__ = ['Shape']

import re
from enum import Enum
from functools import reduce
from itertools import chain, zip_longest
from math import (asin, atan, atan2, ceil, cos, degrees, dist, radians, sin,
                  sqrt)
from typing import (Callable, Dict, Iterable, List, MutableSequence,
                    NamedTuple, NoReturn, Optional, Sequence, SupportsIndex,
                    Tuple, cast, overload)

import numpy as np
from pyquaternion import Quaternion

from .colourspace import Opacity
from .misc import chunk
from .types import Alignment, BézierCoord, CoordinatesView, PropView


class Pixel(NamedTuple):
    """A simple NamedTuple to represent pixels"""
    x: float
    y: float
    alpha: Opacity


class DrawingProp(Enum):
    """
    Enum storing the possible properties of a drawing command.
    Documentation about them is from the Aegisub official one.
    """

    MOVE = 'm'
    """
    m <x> <y> - Move\n
    Moves the cursor to x,y. If you have an unclosed shape, it will automatically be closed,
    as the program assumes that you are now drawing a new, independent shape.
    All drawing routines must start with this command.
    """

    MOVE_NO_CLOSING = 'n'
    """
    n <x> <y> - Move (no closing)\n
    Moves the cursor to x,y, without closing the current shape.
    """

    MOVE_NC = MOVE_NO_CLOSING
    """
    Alias for MOVE_NO_CLOSING
    """

    LINE = 'l'
    """
    l <x> <y> - Line\n
    Draws a line from the current cursor position to x,y, and moves the cursor there afterwards.
    """

    EXTEND_BSPLINE = 'p'
    """
    p <x> <y> - Extend b-spline\n
    Extends the b-spline to x,y. This is essentially the same as adding another pair of coordinates
    at the end of s.
    """

    EX_BSPLINE = EXTEND_BSPLINE
    """
    Alias for EXTEND_BSPLINE
    """
    CUBIC_BÉZIER_CURVE = 'b'
    """
    b <x1> <y1> <x2> <y2> <x3> <y3> - Cubic Bézier curve\n
    Draws a cubic (3rd degree) Bézier curve from the cursor position to (x3,y3),
    using (x1,y1) and (x2,y2) as the control points.
    Check the article on Wikipedia for more information about Bézier curves.
    http://en.wikipedia.org/wiki/B%C3%A9zier_curve
    """

    BÉZIER = CUBIC_BÉZIER_CURVE
    """
    Alias for CUBIC_BÉZIER_CURVE
    """

    CUBIC_BSPLINE = 's'
    """
    s <x1> <y1> <x2> <y2> <x3> <y3> .. <xN> <yN> - Cubic b-spline\n
    Draws a cubic (3rd degree) uniform b-spline to point N.
    This must contain at least 3 coordinates (and is, in that case, the same as b).
    This basically lets you chain several cubic Bézier curves together.
    Check this other article on Wikipedia for more information.
    """

    BSPLINE = CUBIC_BSPLINE
    """
    Alias for CUBIC_BSPLINE
    """

    CLOSE_BSPLINE = 'c'
    """
    c - Close b-spline\n
    Closes the b-spline.
    """

    @classmethod
    def _prop_drawing_dict(cls) -> Dict[str, DrawingProp]:
        return cast(Dict[str, DrawingProp], cls._value2member_map_)


class DrawingCommand(Sequence[Tuple[float, float]]):
    """
    A drawing command is a DrawingProp and a number of coordinates
    """
    _prop: DrawingProp
    _coordinates: List[Tuple[float, float]]

    @property
    def prop(self) -> DrawingProp:
        """The DrawingProp of this DrawingCommand"""
        return self._prop

    @property
    def coordinates(self) -> CoordinatesView[float]:
        """Coordinates of this DrawingCommand"""
        return CoordinatesView(self)

    def __init__(self, prop: DrawingProp, *coordinates: Tuple[float, float]) -> None:
        """
        Make a DrawingCommand object

        :param prop:            Drawing property of this DrawingCommand
        :param coordinates:     Coordinates of this DrawingCommand
        """
        self._prop = prop
        self._coordinates = list(coordinates)
        self.check_integrity()
        super().__init__()

    @overload
    def __getitem__(self, index: SupportsIndex) -> Tuple[float, float]:
        ...

    @overload
    def __getitem__(self, index: slice) -> NoReturn:
        ...

    def __getitem__(self, index: SupportsIndex | slice) -> Tuple[float, float] | NoReturn:
        if isinstance(index, SupportsIndex):
            return self._coordinates[index]
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: slice is not supported!')

    def __len__(self) -> int:
        return len(self._coordinates)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DrawingCommand):
            return NotImplemented
        return o._prop == self._prop and o._coordinates == self._coordinates

    def __str__(self) -> str:
        return self._prop.value + ' ' + ' '.join(f'{x} {y}' for x, y in self)

    def __repr__(self) -> str:
        return repr(str(self._prop) + ', ' + ', '.join(map(str, self)))

    def check_integrity(self) -> None:
        """Check if the current coordinates are valid"""
        DP = DrawingProp
        m, n, l, p = DP.MOVE, DP.MOVE_NC, DP.LINE, DP.EX_BSPLINE
        b, s, c = DP.BÉZIER, DP.CUBIC_BSPLINE, DP.CLOSE_BSPLINE
        if self._prop in {m, n, l, p}:
            check_len = len(self) == 1
        elif self._prop == b:
            check_len = len(self) == 3
        elif self._prop == s:
            check_len = len(self) >= 3
        elif self._prop == c:
            check_len = len(self) == 0
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: Undefined DrawingProp!')
        if not check_len:
            raise ValueError(
                f'{self.__class__.__name__}: "{self._prop}" does not correspond to the length of the coordinates'
                + ''.join(map(str, self))
            )

    def round(self, ndigits: int = 3) -> None:
        """
        Round coordinates to a given precision in decimal digits.

        :param ndigits:         Number of digits
        """
        for i, (x, y) in enumerate(self):
            nx, ny = round(float(x), ndigits), round(float(y), ndigits)
            if (intx := int(nx)) == nx:
                nx = intx
            if (inty := int(ny)) == ny:
                ny = inty
            self._coordinates[i] = nx, ny


class Shape(MutableSequence[DrawingCommand]):
    """
    Class for creating, handling, making transformations from an ASS shape
    """
    _commands: List[DrawingCommand]

    @property
    def props(self) -> PropView[DrawingProp]:
        """The DrawingProp of this DrawingCommand"""
        return PropView([c.prop for c in self])

    @property
    def coordinates(self) -> List[CoordinatesView[float]]:
        """Coordinates of this DrawingCommand"""
        return [c.coordinates for c in self]

    def __init__(self, cmds: Iterable[DrawingCommand]) -> None:
        """
        Initialise a Shape object with given DrawingCommand objects

        :param cmds:        DrawingCommand objects
        """
        self._commands = list(cmds)
        super().__init__()

    @overload
    def __getitem__(self, index: SupportsIndex) -> DrawingCommand:
        ...

    @overload
    def __getitem__(self, index: slice) -> Shape:
        ...

    def __getitem__(self, index: SupportsIndex | slice) -> DrawingCommand | Shape:
        if isinstance(index, SupportsIndex):
            return self._commands[index]
        else:
            return Shape(self._commands[index])

    @overload
    def __setitem__(self, index: SupportsIndex, value: DrawingCommand) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[DrawingCommand]) -> None:
        ...

    def __setitem__(self, index: SupportsIndex | slice, value: DrawingCommand | Iterable[DrawingCommand]) -> None:
        if isinstance(index, SupportsIndex) and isinstance(value, DrawingCommand):
            self._commands[index] = value
        elif isinstance(index, slice) and not isinstance(value, DrawingCommand):
            self._commands[index] = value
        elif isinstance(index, SupportsIndex) and isinstance(value, DrawingCommand):
            raise TypeError(f'{self.__class__.__name__}: can only assign a value!')
        elif isinstance(index, slice) and not isinstance(value, DrawingCommand):
            raise TypeError(f'{self.__class__.__name__}: can only assign an iterable!')
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: not supported')

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        del self._commands[index]

    def __len__(self) -> int:
        return len(self._commands)

    def insert(self, index: int, value: DrawingCommand) -> None:
        """
        Insert a DrawingCommand value before index

        :param index:               Index number
        :param value:               DrawingCommand object
        """
        self._commands.insert(index, value)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, (Shape, str)):
            return NotImplemented
        if isinstance(o, str):
            response = str(self) == o or self.to_str() == o
        else:
            response = all(scmd == so for scmd, so in zip(self, o))
        return response

    def __add__(self, other: Shape) -> Shape:
        new = Shape(self)
        new += other
        return new

    def __iadd__(self, x: Iterable[DrawingCommand]) -> Shape:
        return Shape(super().__iadd__(x))

    def __str__(self) -> str:
        return ' '.join(map(str, self._commands))

    def __repr__(self) -> str:
        return repr(self._commands)

    def to_str(self, round_digits: int = 3, optimise: bool = True) -> str:
        """
        Return the current shape in ASS format

        :param round_digits:    Decimal digits rounding precision, defaults to 3
        :param optimise:        Optimise the string by removing redundant drawing prop, defaults to True
        :return:                Shape in ASS format
        """
        self.round(round_digits)

        if optimise:
            p, draw = DrawingProp.CLOSE_BSPLINE, ''
            for cmd in self:
                if cmd.prop != p:
                    draw += str(cmd) + ' '
                elif cmd.prop == p and cmd.prop in {DrawingProp.LINE, DrawingProp.CUBIC_BÉZIER_CURVE}:
                    draw += ' '.join(f'{x} {y}' for x, y in cmd) + ' '
                else:
                    raise NotImplementedError(f'{self.__class__.__name__}: prop "{cmd.prop} not recognised!"')
                p = cmd.prop
        else:
            draw = str(self)
        return draw

    def round(self, ndigits: int = 3) -> None:
        """
        Round coordinates to a given precision in decimal digits.

        :param ndigits:         Number of digits
        """
        for cmd in self:
            cmd.round(ndigits)

    def map(self, func: Callable[[float, float], Tuple[float, float]]) -> None:
        """
        Sends every point of a shape through given transformation function to change them.

        **Tips:** *Working with outline points can be used to deform the whole shape and make f.e. a wobble effect.*

        :param func:            A function with two parameters representing the x and y coordinates of each point.
                                It will define how each coordinate will be changed.
        """
        for i, cmd in enumerate(self):
            self[i] = DrawingCommand(cmd.prop, *[func(x, y) for x, y in cmd.coordinates])

    def move(self, _x: float = 0., _y: float = 0.) -> None:
        """
        Moves shape coordinates in given direction.

        :param _x:              Displacement along the x-axis, defaults to 0.
        :param _y:              Displacement along the y-axis, defaults to 0.
        """
        self.map(lambda x, y: (x + _x, y + _y))

    def bounding(self) -> Tuple[float, float, float, float]:
        """
        Calculates shape bounding box.

        **Tips:** *Using this you can get more precise information about a shape (width, height, position).*

        Examples:
            ..  code-block:: python3

                x0, y0, x1, y1 = Shape.from_ass_string("m 10 5 l 25 5 25 42 10 42").bounding()
                print(f"Left-top: {x0} {y0}\\nRight-bottom: {x1} {y1}")

            >>> Left-top: 10 5
            >>> Right-bottom: 25 42

        :return:                A tuple of coordinates of the bounding box
        """
        all_x, all_y = set(), set()

        def _func(x: float, y: float) -> Tuple[float, float]:
            all_x.add(x)
            all_y.add(y)
            return x, y

        self.map(_func)
        return min(all_x), min(all_y), max(all_x), max(all_y)

    def align(self, an: Alignment = 7) -> None:
        """
        Automatically align the shape to a given alignment

        :param an:             Alignment argument in the range 1 <= an <= 9
        """
        align = {
            7: (0., 0.),
            8: (-0.5, 0.),
            9: (-1., 0.),
            4: (0., -0.5),
            5: (-0.5, -0.5),
            6: (-1., -0.5),
            1: (0., -1.),
            2: (-0.5, -1.),
            3: (-1., -1.),
        }
        try:
            an_x, an_y = align[an]
        except KeyError as key_err:
            raise ValueError(f'{self.__class__.__name__}: Wrong an value!') from key_err
        x0, y0, x1, y1 = self.bounding()
        self.map(
            lambda x, y:
            (
                x + x0 * -1 + an_x * (x1 + x0 * -1),
                y + y0 * -1 + an_y * (y1 + y0 * -1)
            )
        )

    def rotate_z(self, rotation: float, zero_pad: Optional[Tuple[float, float]] = None) -> None:
        """
        Rotate current shape to a given rotation

        :param rotation:        Rotation in degrees
        :param zero_pad:        Point where the Z-axis rotation will be performed.
                                If not specified, equivalent to (0., 0.), defaults to None
        """
        def _calc(x: float, y: float, zpx: float, zpy: float) -> Tuple[float, float]:
            # Distance to zero-point
            zpd = dist((zpx, zpy), (x, y))

            rot = radians(rotation)
            curot = atan2(x - zpx, y - zpy)

            nx = zpd * sin(curot + rot) + zpx
            ny = zpd * cos(curot + rot) + zpy
            return nx, ny

        self.map(lambda x, y: _calc(x, y, *(0., 0.) if not zero_pad else zero_pad))

        # def _calc(x: float, y: float, zero_pad: Tuple[float, float]) -> Tuple[float, float]:
        #     rot = radians(rotation)
        #     nx = x * cos(rot) + y * sin(rot)
        #     ny = - x * sin(rot) + y * cos(rot)
        #     return nx, ny

    def flatten(self, tolerance: float = 1.) -> None:
        """
        Flatten shape's bezier curves into lines.

        :param tolerance:       Angle in degree to define a curve as flat, defaults to 1.0.
                                Increasing it will boost performance but decrease accuracy.
        """
        # Aliases
        DP = DrawingProp

        m, n, l, p = DP.MOVE, DP.MOVE_NC, DP.LINE, DP.EX_BSPLINE
        b, s, c = DP.BÉZIER, DP.CUBIC_BSPLINE, DP.CLOSE_BSPLINE
        ncmds: List[DrawingCommand] = []

        # Work with the commands reversed
        self.reverse()

        for cmd0, cmd1 in zip_longest(self, self[1:]):
            cmd0, cmd1 = cast(DrawingCommand, cmd0), cast(DrawingCommand, cmd1)
            if cmd0.prop in {m, n, l}:
                ncmds.append(cmd0)
            elif cmd0.prop in {p, s, c}:
                raise NotImplementedError(
                    f'{self.__class__.__name__}: EXTEND_BSPLINE, CUBIC_BSPLINE and CLOSE_BSPLINE'
                    + ' drawing properties are not supported!'
                )
            elif cmd0.prop == b:
                # Get the previous coordinate to complete a bezier curve
                flatten_cmds = self._curve4_to_lines((list(cmd1.coordinates)[-1], *cmd0), tolerance)  # type: ignore
                ncmds.extend(reversed(flatten_cmds))
            else:
                raise NotImplementedError(f'{self.__class__.__name__}: drawing property not recognised!')

        self.clear()
        self.extend(reversed(ncmds))

    def _curve4_to_lines(self, b_coord: BézierCoord, tolerance: float, /) -> List[DrawingCommand]:
        """function to convert 4th degree curve to line points"""

        ncoord: List[DrawingCommand] = []

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
            rvecsp = vecsp[::-1]
            for v0, v1 in reversed(list(zip(rvecsp[1:], rvecsp))):
                if abs(self._get_vector_angle(v0, v1)) > tolerance:
                    return False

            return True

        def _convert_recursive(b_coord: BézierCoord, /) -> None:
            """Conversion in recursive processing"""
            if _curve4_is_flat(b_coord):
                ncoord.append(
                    DrawingCommand(DrawingProp.LINE, b_coord[-1])
                )
                return None
            b0, b1 = _curve4_subdivide(b_coord)
            _convert_recursive(b0)
            _convert_recursive(b1)

        # Splitting curve recursively until we're not satisfied (angle <= tolerance)
        _convert_recursive(b_coord)
        return ncoord

    @staticmethod
    def _get_vector_angle(v0: Tuple[float, float], v1: Tuple[float, float]) -> float:
        """Get angle between two vectors"""
        # https://stackoverflow.com/a/35134034
        (x0, y0), (x1, y1) = v0, v1

        angler = atan2(np.linalg.det((v0, v1)), np.dot(v0, v1))
        angled = degrees(angler)

        # Return with sign by clockwise direction
        return - angled if (x0 * y1 - y0 * x1) < 0 else angled

    @staticmethod
    def _get_vector_length(vector: Tuple[float, float]) -> float:
        # Not used btw
        return np.linalg.norm(vector)

    def split(self, max_length: float = 16., tolerance: float = 1.) -> None:
        """
        Flatten Shape bezier curves into lines and split the latter into shorter segments
        with maximum given length.

        :param max_len:         Maximum length between two consecutive points, defaults to 16.
        :param tolerance:       Angle in degree to define a curve as flat, defaults to 1.0.
                                Increasing it will boost performance but decrease accuracy.
        """
        # Aliases
        DP = DrawingProp
        m, n, l = DP.MOVE, DP.MOVE_NC, DP.LINE
        ncmds: List[DrawingCommand] = []

        self.flatten(tolerance)

        # Work with the commands reversed
        self.reverse()

        for cmd0, cmd1 in zip_longest(self, self[1:]):
            cmd0, cmd1 = cast(DrawingCommand, cmd0), cast(DrawingCommand, cmd1)
            if cmd0.prop in {m, n}:
                ncmds.append(cmd0)
            elif cmd0.prop == l:
                # Get the new points
                splitted_cmds = self._split_line(cmd1._coordinates[-1], cmd0._coordinates[0], max_length)
                ncmds.extend(reversed(splitted_cmds))
            else:
                raise NotImplementedError(f'{self.__class__.__name__}: drawing property not recognised!')

        self.clear()
        self.extend(reversed(ncmds))

    def _split_line(self, p0: Tuple[float, float], p1: Tuple[float, float], max_length: float) -> List[DrawingCommand]:
        """Split a vector (p0, p1) into shorter vectors with maximum max_length"""
        l = DrawingProp.LINE
        ncmds: List[DrawingCommand] = []

        distance = dist(p0, p1)
        if distance > max_length:
            (x0, y0), (x1, y1) = p0, p1
            # Equal step between the two points instead of having all points to 16
            # except the last for the remaining distance
            step = distance / ceil(distance / max_length)
            # Step can be a float so numpy.arange is prefered
            for i in np.arange(step, distance, step):
                pct = i / distance
                ncmds.append(
                    DrawingCommand(l, ((x0 + (x1 - x0) * pct), (y0 + (y1 - y0) * pct)))
                )
        ncmds.append(DrawingCommand(l, p1))
        return ncmds

    @classmethod
    def ring(cls, out_rad: float, in_rad: float, c_xy: Tuple[float, float] = (0., 0.), /) -> Shape:
        """
        Make a ring Shape object with given inner and outer radius, centered around (c_xy)

        **Tips:** *A ring with increasing inner radius, starting from 0, can look like an outfading point.*

        :param out_rad:         Outer radius for the ring
        :param in_rad:          Inner radius for the ring
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :return:                A Shape object representing a ring
        """
        if out_rad <= in_rad:
            raise ValueError(f'{cls.__name__}: inner radius must be less than outer radius')

        return cls.disk(out_rad, c_xy, True) + cls.disk(in_rad, c_xy, False)

    @classmethod
    def disk(cls, radius: float, c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True) -> Shape:
        """
        Make a disk Shape object with given radius, centered around (c_xy)

        :param radius:          Radius of the disk
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :return:                A Shape object representing a disk
        """
        return cls.ellipse(radius, radius, c_xy, clockwise)

    @classmethod
    def ellipse(cls, w: float, h: float, c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True) -> Shape:
        """
        Make an ellipse Shape object with given width and height, centered around (c_xy)

        :param w:               Width of the ellipse
        :param h:               Height of the ellipse
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :return:                A Shape object representing an ellipse
        """
        c = 0.551915024494  # https://spencermortensen.com/articles/bezier-circle/
        cx, cy = c_xy

        # Aliases
        DP = DrawingCommand
        m, b = DrawingProp.MOVE, DrawingProp.BÉZIER

        cl = - int((-1) ** clockwise)
        cmds = [
            DP(m, ((cx - 0) * cl, cy + h)),  # Start from bottom center
            DP(
                b,
                ((cx - w * c) * cl, cy + h),
                ((cx - w) * cl, cy + h * c),
                ((cx - w) * cl, cy - 0)
            ),
            DP(
                b,
                ((cx - w) * cl, cy - h * c),
                ((cx - w * c) * cl, cy - h),
                ((cx - 0) * cl, cy - h)
            ),
            DP(
                b,
                ((cx + w * c) * cl, cy - h),
                ((cx + w) * cl, cy - h * c),
                ((cx + w) * cl, cy - 0)
            ),
            DP(
                b,
                ((cx + w) * cl, cy + h * c),
                ((cx + w * c) * cl, cy + h),
                ((cx - 0) * cl, cy + h)
            )
        ]
        return cls(cmds)

    @classmethod
    def heart(cls, size: float = 30., c_xy: Tuple[float, float] = (0., 0.), voffset: float = 0., /) -> Shape:
        """
        Make an heart Shape object with given size

        :param size:            Size of the heart, defaults to 30.
        :param voffset:         Vertical offset of the central coordinate, defaults to 0.
        :return:                A Shape object representing an heart
        """
        mult = size / 30
        DC = DrawingCommand
        m, b = DrawingProp.MOVE, DrawingProp.BÉZIER

        cmds = [
            DC(m, (15 * mult, 30 * mult)),
            DC(b, (27 * mult, 22 * mult), (30 * mult, 18 * mult), (30 * mult, 14 * mult)),
            DC(b, (31 * mult, 7 * mult), (22 * mult, 0), (15 * mult, 10 * mult + voffset)),
            DC(b, (8 * mult, 0), (-1 * mult, 7 * mult), (0, 14 * mult)),
            DC(b, (0, 18 * mult), (3 * mult, 22 * mult), (15 * mult, 30 * mult))
        ]
        return cls(cmds)

    @classmethod
    def square(cls, length: float, c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True) -> Shape:
        """
        Make a square Shape object with given width, centered around (c_xy)

        :param length:          Length of the square
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :return:                A Shape object representing a square
        """
        return cls.rectangle(length, length, c_xy, clockwise)

    @classmethod
    def rectangle(cls, w: float, h: float, c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True) -> Shape:
        """
        Make a rectangle Shape object with given width, centered around (c_xy)

        :param w:               Width of the rectangle
        :param h:               Height of the rectangle
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :return:                A Shape object representing a rectangle
        """
        return cls.parallelogram(w, h, 90, c_xy, clockwise)

    @classmethod
    def diamond(cls, length: float, angle: float, c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True) -> Shape:
        """
        Make a diamond Shape object with given length and angle, centered around (c_xy)

        :param length:          Length of the diamond
        :param angle:           First angle of the diamond in degrees
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :return:                A Shape object representing a diamond
        """
        return cls.parallelogram(length, length / cos(radians(90 - angle)), angle, c_xy, clockwise)

    @classmethod
    def parallelogram(cls, w: float, h: float, angle: float, c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True) -> Shape:
        """
        Make a parallelogram Shape object with given width, height and angle, centered around (c_xy)

        :param w:               Width of the parallelogram
        :param h:               Height of the parallelogram
        :param angle:           First angle of the parallelogram in degrees
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :return:                A Shape object representing a parallelogram
        """
        DC = DrawingCommand
        mov, lin = DrawingProp.MOVE, DrawingProp.LINE
        cl = - int((-1) ** clockwise)
        cx, cy = c_xy

        l = h / cos(radians(90 - angle))
        x0, y0 = 0, 0
        x1, y1 = l * cos(radians(angle)), l * sin(radians(angle))
        x2, y2 = x1 + w, y1
        x3, y3 = w, 0

        cmds = [
            DC(mov, ((x0 + cx) * cl, y0 + cy)),
            DC(lin, ((x1 + cx) * cl, y1 + cy)),
            DC(lin, ((x2 + cx) * cl, y2 + cy)),
            DC(lin, ((x3 + cx) * cl, y3 + cy)),
            DC(lin, ((x0 + cx) * cl, y0 + cy))
        ]
        return cls(cmds)


    @classmethod
    def equilateral_tr(cls, height: float, c_xy: Tuple[float, float] = (0., 0.), /,
                       clockwise: bool = True, *, orthocentred: bool = True) -> Shape:
        """
        Make a equilateral triangle Shape object with given height, centered around (c_xy)

        :param height:          Height of the triangle
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :param orthocentred:    Centred in the orthocenter, defaults to True
        :return:                A Shape object representing a equilateral triangle
        """
        return cls.triangle(height * 2 / sqrt(3), (60, 60), c_xy, clockwise, orthocentred=orthocentred)

    @classmethod
    def isosceles_tr(cls, height: float, base: float, c_xy: Tuple[float, float] = (0., 0.), /,
                     clockwise: bool = True, *, orthocentred: bool = True) -> Shape:
        """
        Make a isosceles triangle Shape object with given height and base, centered around (c_xy)

        :param height:          Height of the triangle
        :param base:            Lenght of the base of the triangle
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :param orthocentred:    Centred in the orthocenter, defaults to True
        :return:                A Shape object representing a isosceles triangle
        """
        angle = degrees(atan(height / (base / 2)))
        return cls.triangle(base, (angle, angle), c_xy, clockwise, orthocentred=orthocentred)

    @classmethod
    def orthogonal_tr(cls, side: Tuple[float, float], c_xy: Tuple[float, float] = (0., 0.), /,
                      clockwise: bool = True, *, orthocentred: bool = True) -> Shape:
        """
        Make an orthognal (right-angled) triangle Shape object with given sides, centered around (c_xy)

        :param side:            First two sides of the triangle
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :param orthocentred:    Centred in the orthocenter, defaults to True
        :return:                A Shape object representing an orthognal triangle
        """
        return cls.triangle(side, 90, c_xy, clockwise, orthocentred=orthocentred)

    @overload
    @classmethod
    def triangle(cls, side: float, angle: Tuple[float, float], c_xy: Tuple[float, float] = (0., 0.), /,
                 clockwise: bool = True, *, orthocentred: bool = True) -> Shape:
        """
        Make a general triangle Shape object with given side and angles, centered around (c_xy)

        :param side:            First side of the triangle
        :param angle:           First two angles of the triangle in degrees
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :param orthocentred:    Centred in the orthocenter, defaults to True
        :return:                A Shape object representing a triangle
        """
        ...

    @overload
    @classmethod
    def triangle(cls, side: Tuple[float, float], angle: float, c_xy: Tuple[float, float] = (0., 0.), /,
                 clockwise: bool = True, *, orthocentred: bool = True) -> Shape:
        """
        Make a general triangle Shape object with given sides and angle, centered around (c_xy)

        :param side:            First two sides of the triangle
        :param angle:           First angle of the triangle in degrees
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :param orthocentred:    Centred in the orthocenter, defaults to True
        :return:                A Shape object representing a triangle
        """
        ...

    @classmethod
    def triangle(cls, side: float | Tuple[float, float], angle: Tuple[float, float] | float,
                 c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True, *,
                 orthocentred: bool = True) -> Shape:
        DC = DrawingCommand
        mov, lin = DrawingProp.MOVE, DrawingProp.LINE
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
            raise ValueError(f'{cls.__name__}: possibles values are one side and two angles or two sides and one angle')

        cmds = [
            DC(mov, ((x0 + cx) * cl, y0 + cy)),
            DC(lin, ((x1 + cx) * cl, y1 + cy)),
            DC(lin, ((x2 + cx) * cl, y2 + cy)),
            DC(lin, ((x0 + cx) * cl, y0 + cy)),
        ]

        triangle = cls(cmds)

        if orthocentred:
            _, yb0, _, yb1 = triangle.bounding()
            triangle.move(0, (yb1 - yb0) / 6)

        return triangle

    @classmethod
    def from_ass_string(cls, drawing_cmds: str) -> Shape:
        """
        Make a Shape object from a drawing command string in ASS format

        :param drawing_cmds:    String of drawing commands
        :return:                Shape object
        """
        if not drawing_cmds.startswith('m'):
            raise ValueError(f'{cls.__name__}: a shape must have a "m" at the beginning!')

        DC, DP = DrawingCommand, DrawingProp
        cmds: List[DrawingCommand] = []
        draws = cast(List[str], re.findall(r'[mnlpbsc][^mnlpbsc]+', drawing_cmds))

        for draw in draws:
            sdraw = draw.split()
            lendraw = len(sdraw)
            if sdraw[0].startswith(('m', 'n')) and lendraw == 3:
                cmds.append(
                    DC(
                        DP._prop_drawing_dict()[sdraw.pop(0)],
                        (float(sdraw.pop(0)), float(sdraw.pop(0)))
                    )
                )
            elif sdraw[0].startswith('c') and lendraw == 1:
                cmds.append(DC(DP.CLOSE_BSPLINE))
            elif sdraw[0].startswith(('l', 'p')) and lendraw >= 3:
                p = sdraw.pop(0)
                cmds.extend(
                    DC(
                        DP._prop_drawing_dict()[p],
                        (float(x), float(y))
                    ) for x, y in chunk(sdraw, 2)
                )
            elif sdraw[0].startswith(('b', 's')) and lendraw >= 7:
                p = sdraw.pop(0)
                if p == 'b' and (lendraw - 1) / 2 % 3 == 0.:
                    cmds.extend(
                        DC(
                            DP.CUBIC_BÉZIER_CURVE, *coords
                        ) for coords in chunk(chunk(map(float, sdraw), 2), 3)
                    )
                elif p == 's':
                    coords = list(chunk(map(float, sdraw), 2))
                    cmds.append(DC(DP.CUBIC_BSPLINE, *coords))
                else:
                    raise ValueError(f'{cls.__name__}: "{p}" and "{sdraw}" not recognised!')
            else:
                raise ValueError(f'{cls.__name__}: unexpected shape "{draw}"!')
        return cls(cmds)


class OldShape:
    """
    This class can be used to define a Shape object (by passing its drawing commands)
    and then apply functions to it in order to accomplish some tasks, like analyzing its bounding box, apply transformations,
    splitting curves into segments...

    Args:
        drawing_cmds (str): The shape's drawing commands in ASS format as a string.
    """

    # def flatten(self, tolerance: float = 1.0) -> OldShape:
    #     """Splits shape's bezier curves into lines.

    #     | This is a low level function. Instead, you should use :func:`split` which already calls this function.

    #     Parameters:
    #         tolerance (float): Angle in degree to define a curve as flat (increasing it will boost performance during reproduction,
    #         but lower accuracy)

    #     Returns:
    #         A pointer to the current object.

    #     Returns:
    #         The shape as a string, with bezier curves converted to lines.
    #     """
    #     # TO DO: Make this function iterative, recursion is bad.
    #     if tolerance < 0:
    #         raise ValueError("Tolerance must be a positive number")

    #     # Inner functions definitions
    #     # 4th degree curve subdivider (De Casteljau)
    #     def curve4_subdivide(x0, y0, x1, y1, x2, y2, x3, y3, pct):
    #         # Calculate points on curve vectors
    #         x01, y01, x12, y12, x23, y23 = (
    #             (x0 + x1) * pct,
    #             (y0 + y1) * pct,
    #             (x1 + x2) * pct,
    #             (y1 + y2) * pct,
    #             (x2 + x3) * pct,
    #             (y2 + y3) * pct,
    #         )
    #         x012, y012, x123, y123 = (
    #             (x01 + x12) * pct,
    #             (y01 + y12) * pct,
    #             (x12 + x23) * pct,
    #             (y12 + y23) * pct,
    #         )
    #         x0123, y0123 = (x012 + x123) * pct, (y012 + y123) * pct
    #         # Return new 2 curves
    #         return x0, y0, x01, y01, x012, y012, x0123, y0123, x0123, y0123, x123, y123, x23, y23, x3, y3

    #     # Check flatness of 4th degree curve with angles
    #     def curve4_is_flat(x0, y0, x1, y1, x2, y2, x3, y3):
    #         # Pack curve vectors (only ones non zero)
    #         vecs = [[x1 - x0, y1 - y0], [x2 - x1, y2 - y1], [x3 - x2, y3 - y2]]
    #         vecs = [el for el in vecs if not (el[0] == 0 and el[1] == 0)]

    #         # Inner functions to calculate degrees between two 2d vectors
    #         def dotproduct(v1, v2):
    #             return sum((a * b) for a, b in zip(v1, v2))

    #         def length(v):
    #             return math.sqrt(dotproduct(v, v))

    #         def get_angle(v1, v2):
    #             calc = max(
    #                 min(dotproduct(v1, v2) / (length(v1) * length(v2)), 1), -1
    #             )  # Clamping value to prevent errors
    #             angle = math.degrees(math.acos(calc))
    #             if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
    #                 return -angle
    #             return angle

    #         # Check flatness on vectors
    #         for i in range(1, len(vecs)):
    #             if abs(get_angle(vecs[i - 1], vecs[i])) > tolerance:
    #                 return False
    #         return True

    #     # Inner function to convert 4th degree curve to line points
    #     def curve4_to_lines(x0,y0,x1,y1,x2,y2,x3,y3):
    #         # Line points buffer
    #         pts = ""

    #         # Conversion in recursive processing
    #         def convert_recursive(x0, y0, x1, y1, x2, y2, x3, y3):
    #             if curve4_is_flat(x0, y0, x1, y1, x2, y2, x3, y3):
    #                 nonlocal pts
    #                 x3, y3 = Shape.format_value(x3), Shape.format_value(y3)
    #                 pts += f"{x3} {y3} "
    #                 return

    #             x10, y10, x11, y11, x12, y12, x13, y13, x20, y20, x21, y21, x22, y22, x23, y23 = curve4_subdivide(x0, y0, x1, y1, x2, y2, x3, y3, 0.5)
    #             convert_recursive(x10, y10, x11, y11, x12, y12, x13, y13)
    #             convert_recursive(x20, y20, x21, y21, x22, y22, x23, y23)

    #         # Splitting curve recursively until we're not satisfied (angle <= tolerance)
    #         convert_recursive(x0, y0, x1, y1, x2, y2, x3, y3)
    #         # Return resulting points
    #         return " ".join(pts[:-1].split(" ")[:-2])  # Delete last space and last two float values

    #     # Getting all points and commands in a list
    #     cmds_and_points = self.drawing_cmds.split()
    #     i = 0
    #     n = len(cmds_and_points)

    #     # Scanning all commands and points (improvable)
    #     while i < n:
    #         if cmds_and_points[i] == "b":  # We've found a curve, let's split it into lines
    #             try:
    #                 # Getting all the points: if we don't have exactly 8 points, shape is not valid
    #                 x0, y0 = (
    #                     float(cmds_and_points[i - 2]),
    #                     float(cmds_and_points[i - 1]),
    #                 )
    #                 x1, y1 = (
    #                     float(cmds_and_points[i + 1]),
    #                     float(cmds_and_points[i + 2]),
    #                 )
    #                 x2, y2 = (
    #                     float(cmds_and_points[i + 3]),
    #                     float(cmds_and_points[i + 4]),
    #                 )
    #                 x3, y3 = (
    #                     float(cmds_and_points[i + 5]),
    #                     float(cmds_and_points[i + 6]),
    #                 )
    #             except IndexError:
    #                 raise ValueError(
    #                     "Shape providen is not valid (not enough points for a curve)"
    #                 )

    #             # Obtaining the converted curve and saving it for later
    #             cmds_and_points[i] = "l"
    #             cmds_and_points[i + 1] = curve4_to_lines(x0, y0, x1, y1, x2, y2, x3, y3)

    #             i += 2
    #             n -= 3

    #             # Deleting the remaining points
    #             for _ in range(3):
    #                 del cmds_and_points[i]

    #             # Going to the next point
    #             i += 2

    #             # Check if we're at the end of the shape
    #             if i < n:
    #                 # Check for implicit bezier curve
    #                 try:
    #                     float(cmds_and_points[i])  # Next number is a float?
    #                     cmds_and_points.insert(i, "b")
    #                     n += 1
    #                 except ValueError:
    #                     pass
    #         elif cmds_and_points[i] == "c":  # Deleting c tag?
    #             del cmds_and_points[i]
    #             n -= 1
    #         else:
    #             i += 1

    #     # Update shape
    #     self.drawing_cmds = " ".join(cmds_and_points)
    #     return self

    # def split(self, max_len: float = 16, tolerance: float = 1.0) -> OldShape:
    #     """Splits shape bezier curves into lines and splits lines into shorter segments with maximum given length.

    #     **Tips:** *You can call this before using :func:`map` to work with more outline points for smoother deforming.*

    #     Parameters:
    #         max_len (int or float):
    #             The max length that you want all the lines to be

    #         tolerance (float):
    #             Angle in degree to define a bezier curve as flat. Currently not implemented.
    #             (increasing it will boost performance during reproduction, but lower accuracy)

    #     Returns:
    #         A pointer to the current object.

    #     Examples:
    #         ..  code-block:: python3

    #             print( Shape("m -100.5 0 l 100 0 b 100 100 -100 100 -100.5 0 c").split() )

    #         >>> m -100.5 0 l -100 0 -90 0 -80 0 -70 0 -60 0 -50 0 -40 0 -30 0 -20 0 -10 0 0 0 10 0 20 0 30 0 40 0 50 0 60 0 70 0 80 0 90 0 100 0 l 99.964 2.325 99.855 4.614 99.676 6.866 99.426 9.082 99.108 11.261 98.723 13.403 98.271 15.509 97.754 17.578 97.173 19.611 96.528 21.606 95.822 23.566 95.056 25.488 94.23 27.374 93.345 29.224 92.403 31.036 91.405 32.812 90.352 34.552 89.246 36.255 88.086 37.921 86.876 39.551 85.614 41.144 84.304 42.7 82.945 44.22 81.54 45.703 80.88 47.15 78.592 48.56 77.053 49.933 75.471 51.27 73.848 52.57 72.184 53.833 70.482 55.06 68.742 56.25 66.965 57.404 65.153 58.521 63.307 59.601 61.427 60.645 59.515 61.652 57.572 62.622 55.599 63.556 53.598 64.453 51.569 65.314 49.514 66.138 47.433 66.925 45.329 67.676 43.201 68.39 41.052 69.067 38.882 69.708 36.692 70.312 34.484 70.88 32.259 71.411 27.762 72.363 23.209 73.169 18.61 73.828 13.975 74.341 9.311 74.707 4.629 74.927 -0.62 75 -4.755 74.927 -9.438 74.707 -14.103 74.341 -18.741 73.828 -23.343 73.169 -27.9 72.363 -32.402 71.411 -34.63 70.88 -36.841 70.312 -39.033 69.708 -41.207 69.067 -43.359 68.39 -45.49 67.676 -47.599 66.925 -49.683 66.138 -51.743 65.314 -53.776 64.453 -55.782 63.556 -57.759 62.622 -59.707 61.652 -61.624 60.645 -63.509 59.601 -65.361 58.521 -67.178 57.404 -68.961 56.25 -70.707 55.06 -72.415 53.833 -74.085 52.57 -75.714 51.27 -77.303 49.933 -78.85 48.56 -80.353 47.15 -81.811 45.703 -83.224 44.22 -84.59 42.7 -85.909 41.144 -87.178 39.551 -88.397 37.921 -89.564 36.255 -90.68 34.552 -91.741 32.812 -92.748 31.036 -93.699 29.224 -94.593 27.374 -95.428 25.488 -96.205 23.566 -96.92 21.606 -97.575 19.611 -98.166 17.578 -98.693 15.509 -99.156 13.403 -99.552 11.261 -99.881 9.082 -100.141 6.866 -100.332 4.614 -100.452 2.325 -100.5 0
    #     """
    #     if max_len <= 0:
    #         raise ValueError(
    #             "The length of segments must be a positive and non-zero value"
    #         )

    #     # Internal function to help splitting a line
    #     def line_split(x0: float, y0: float, x1: float, y1: float):
    #         x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    #         # Line direction & length
    #         rel_x, rel_y = x1 - x0, y1 - y0
    #         distance = math.sqrt(rel_x * rel_x + rel_y * rel_y)
    #         # If the line is too long -> split
    #         if distance > max_len:
    #             lines: list[str] = []
    #             distance_rest = distance % max_len
    #             cur_distance = distance_rest if distance_rest > 0 else max_len

    #             while cur_distance <= distance:
    #                 pct = cur_distance / distance
    #                 x, y = (
    #                     Shape.format_value(x0 + rel_x * pct),
    #                     Shape.format_value(y0 + rel_y * pct),
    #                 )

    #                 lines.append(f"{x} {y}")
    #                 cur_distance += max_len

    #             return " ".join(lines), lines[-1].split()
    #         else:  # No line split
    #             x1_str, y1_str = Shape.format_value(x1), Shape.format_value(y1)
    #             return f"{x1_str} {y1_str}", [x1_str, y1_str]

    #     # Getting all points and commands in a list
    #     cmds_and_points = self.flatten().drawing_cmds.split()
    #     i = 0
    #     n = len(cmds_and_points)

    #     # Utility variables
    #     is_line = False
    #     previous_two = None
    #     last_move = None

    #     # Splitting everything splittable, probably improvable
    #     while i < n:
    #         current = cmds_and_points[i]
    #         if current == "l":
    #             # Activate line mode, save previous two points
    #             is_line = True
    #             if (
    #                 not previous_two
    #             ):  # If we're not running into contiguous line, we need to save the previous two
    #                 previous_two = [cmds_and_points[i - 2], cmds_and_points[i - 1]]
    #             i += 1
    #         elif (
    #             current == "m"
    #             or current == "n"
    #             or current == "b"
    #             or current == "s"
    #             or current == "p"
    #             or current == "c"
    #         ):
    #             if current == "m":
    #                 if (
    #                     last_move
    #                 ):  # If we had a previous move, we need to close the previous figure before proceding
    #                     x0, y0 = None, None
    #                     if previous_two:  # If I don't have previous point, I can read them on cmds_and_points, else I wil take 'em
    #                         x0, y0 = previous_two[0], previous_two[1]
    #                     else:
    #                         x0, y0 = cmds_and_points[i - 2], cmds_and_points[i - 1]

    #                     if not (
    #                         x0 == last_move[0] and y0 == last_move[1]
    #                     ):  # Closing last figure
    #                         cmds_and_points[i] = (
    #                             line_split(x0, y0, last_move[0], last_move[1])[0] + " m"
    #                         )
    #                 last_move = [cmds_and_points[i + 1], cmds_and_points[i + 2]]

    #             # Disabling line mode, removing previous two points
    #             is_line = False
    #             previous_two = None
    #             i += 1
    #         elif is_line:
    #             # Do the work with the two points found and the previous two
    #             cmds_and_points[i], previous_two = line_split(
    #                 previous_two[0],
    #                 previous_two[1],
    #                 cmds_and_points[i],
    #                 cmds_and_points[i + 1],
    #             )
    #             del cmds_and_points[i + 1]
    #             # Let's go to the next two points or tag
    #             i += 1
    #             n -= 1
    #         else:  # We're working with points that are not lines points, let's go forward
    #             i += 2

    #     # Close last figure of new shape, taking two last points and two last points of move
    #     i = n
    #     if not previous_two:
    #         while i >= 0:
    #             current = cmds_and_points[i]
    #             current_prev = cmds_and_points[i - 1]
    #             if (
    #                 current != "m"
    #                 and current != "n"
    #                 and current != "b"
    #                 and current != "s"
    #                 and current != "p"
    #                 and current != "c"
    #                 and current_prev != "m"
    #                 and current_prev != "n"
    #                 and current_prev != "b"
    #                 and current_prev != "s"
    #                 and current_prev != "p"
    #                 and current_prev != "c"
    #             ):
    #                 previous_two = [current, current_prev]
    #                 break
    #             i -= 1
    #     if not previous_two[0] == last_move[0] and previous_two[1] == last_move[1]:  # Split!
    #         cmds_and_points.append(
    #             "l "
    #             + line_split(
    #                 previous_two[0], previous_two[1], last_move[0], last_move[1]
    #             )[0]
    #         )

    #     # Sew up everything back and update shape
    #     self.drawing_cmds = " ".join(cmds_and_points)
    #     return self

    def __to_outline(
        self, bord_xy: float, bord_y: float = None, mode: str = "round"
    ) -> OldShape:
        """Converts shape command for filling to a shape command for stroking.

        **Tips:** *You could use this for border textures.*

        Parameters:
            shape (str): The shape in ASS format as a string.

        Returns:
            A pointer to the current object.

        Returns:
            A new shape as string, representing the border of the input.
        """
        raise NotImplementedError

    @classmethod
    def __glance_or_star(cls, edges: int, inner_size: float, outer_size: float, g_or_s: str) -> OldShape:
        """
        General function to create a shape object representing star or glance.
        """
        # Alias for utility functions
        f = cls.format_value

        def rotate_on_axis_z(point, theta):
            # Internal function to rotate a point around z axis by a given angle.
            theta = math.radians(theta)
            return Quaternion(axis=[0, 0, 1], angle=theta).rotate(point)

        # Building shape
        shape = ["m 0 %s %s" % (-outer_size, g_or_s)]
        inner_p, outer_p = 0, 0

        for i in range(1, edges + 1):
            # Inner edge
            inner_p = rotate_on_axis_z([0, -inner_size, 0], ((i - 0.5) / edges) * 360)
            # Outer edge
            outer_p = rotate_on_axis_z([0, -outer_size, 0], (i / edges) * 360)
            # Add curve / line
            if g_or_s == "l":
                shape.append(
                    "%s %s %s %s"
                    % (f(inner_p[0]), f(inner_p[1]), f(outer_p[0]), f(outer_p[1]))
                )
            else:
                shape.append(
                    "%s %s %s %s %s %s"
                    % (
                        f(inner_p[0]),
                        f(inner_p[1]),
                        f(inner_p[0]),
                        f(inner_p[1]),
                        f(outer_p[0]),
                        f(outer_p[1]),
                    )
                )

        shape = cls(" ".join(shape))
        shape.move()

        # Return result centered
        return shape

    @classmethod
    def star(cls, edges: int, inner_size: float, outer_size: float) -> OldShape:
        """Returns a shape object of a star object with given number of outer edges and sizes, centered around (0,0).

        **Tips:** *Different numbers of edges and edge distances allow individual n-angles.*

        Parameters:
            edges (int): The number of edges of the star.
            inner_size (int or float): The inner edges distance from center.
            outer_size (int or float): The outer edges distance from center.

        Returns:
            A shape object as a string representing a star.
        """
        return cls.__glance_or_star(edges, inner_size, outer_size, "l")

    @classmethod
    def glance(cls, edges: int, inner_size: float, outer_size: float) -> OldShape:
        """Returns a shape object of a glance object with given number of outer edges and sizes, centered around (0,0).

        **Tips:** *Glance is similar to Star, but with curves instead of inner edges between the outer edges.*

        Parameters:
            edges (int): The number of edges of the star.
            inner_size (int or float): The inner edges distance from center.
            outer_size (int or float): The control points for bezier curves between edges distance from center.

        Returns:
            A shape object as a string representing a glance.
        """
        return cls.__glance_or_star(edges, inner_size, outer_size, "b")

    def to_pixels(self, supersampling: int = 8) -> List[Pixel]:
        """| Converts a Shape object to a list of pixel data.
        | A pixel data is a NamedTuple with the attributes 'x' (horizontal position), 'y' (vertical position) and 'alpha' (alpha/transparency/opacity).

        It is highly suggested to create a dedicated style for pixels,
        because you will write less tags for line in your pixels, which means less size for your .ass file.

        | The style suggested is:
        | - **an=7 (very important!);**
        | - bord=0;
        | - shad=0;
        | - For Font informations leave whatever the default is;

        **Tips:** *As for text, even shapes can decay!*

        Parameters:
            shape (Shape): An object of class Shape.
            supersampling (int): Value used for supersampling. Higher value means smoother and more precise anti-aliasing (and more computational time for generation).

        Returns:
            A list of dictionaries representing each individual pixel of the input shape.

        Examples:
            ..  code-block:: python3

                line = lines[2].copy()
                line.style = "p"
                p_sh = Shape.rectangle()
                for pixel in Convert.shape_to_pixels(Shape.heart(100)):
                    # Random circle to pixel effect just to show
                    x, y = math.floor(line.left) + pixel.x, math.floor(line.top) + pixel.y
                    alpha = "\\alpha" + Convert.color_alpha_to_ass(pixel.alpha) if pixel.alpha != 255 else ""

                    line.text = "{\\p1\\pos(%d,%d)%s\\fad(0,%d)}%s" % (x, y, alpha, l.dur/4, p_sh)
                    io.write_line(line)
        """
        # Scale values for supersampled rendering
        upscale = supersampling
        downscale = 1 / upscale

        # Upscale shape for later downsampling
        self.map(lambda x, y: (x * upscale, y * upscale))

        # Bring shape near origin in positive room
        x1, y1, x2, y2 = self.bounding()
        shift_x, shift_y = -1 * (x1 - x1 % upscale), -1 * (y1 - y1 % upscale)
        self.move(shift_x, shift_y)

        # Create image
        width, height = (
            math.ceil((x2 + shift_x) * downscale) * upscale,
            math.ceil((y2 + shift_y) * downscale) * upscale,
        )
        image: List[bool] = [False] * (width * height)

        # Renderer (on binary image with aliasing)
        lines, last_point, last_move = [], {}, {}

        def collect_lines(x, y, typ):
            # Collect lines (points + vectors)
            nonlocal lines, last_point, last_move
            x, y = int(round(x)), int(round(y))  # Use integers to avoid rounding errors

            # Move
            if typ == "m":
                # Close figure with non-horizontal line in image
                if (
                    last_move
                    and last_move["y"] != last_point["y"]
                    and not (last_point["y"] < 0 and last_move["y"] < 0)
                    and not (last_point["y"] > height and last_move["y"] > height)
                ):
                    lines.append(
                        [
                            last_point["x"],
                            last_point["y"],
                            last_move["x"] - last_point["x"],
                            last_move["y"] - last_point["y"],
                        ]
                    )

                last_move = {"x": x, "y": y}
            # Non-horizontal line in image
            elif (
                last_point
                and last_point["y"] != y
                and not (last_point["y"] < 0 and y < 0)
                and not (last_point["y"] > height and y > height)
            ):
                lines.append(
                    [
                        last_point["x"],
                        last_point["y"],
                        x - last_point["x"],
                        y - last_point["y"],
                    ]
                )

            # Remember last point
            last_point = {"x": x, "y": y}

        self.flatten().map(collect_lines)

        # Close last figure with non-horizontal line in image
        if (
            last_move
            and last_move["y"] != last_point["y"]
            and not (last_point["y"] < 0 and last_move["y"] < 0)
            and not (last_point["y"] > height and last_move["y"] > height)
        ):
            lines.append(
                [
                    last_point["x"],
                    last_point["y"],
                    last_move["x"] - last_point["x"],
                    last_move["y"] - last_point["y"],
                ]
            )

        # Calculates line x horizontal line intersection
        def line_x_hline(x, y, vx, vy, y2):
            if vy != 0:
                s = (y2 - y) / vy
                if s >= 0 and s <= 1:
                    return x + s * vx
            return None

        # Scan image rows in shape
        _, y1, _, y2 = self.bounding()
        for y in range(max(math.floor(y1), 0), min(math.ceil(y2), height)):
            # Collect row intersections with lines
            row_stops = []
            for line in lines:
                cx = line_x_hline(line[0], line[1], line[2], line[3], y + 0.5)
                if cx is not None:
                    row_stops.append(
                        [max(0, min(cx, width)), 1 if line[3] > 0 else -1]
                    )  # image trimmed stop position & line vertical direction

            # Enough intersections / something to render?
            if len(row_stops) > 1:
                # Sort row stops by horizontal position
                row_stops.sort(key=lambda x: x[0])
                # Render!
                status, row_index = 0, y * width
                for i in range(0, len(row_stops) - 1):
                    status = status + row_stops[i][1]
                    if status != 0:
                        for x in range(
                            math.ceil(row_stops[i][0] - 0.5),
                            math.floor(row_stops[i + 1][0] + 0.5),
                        ):
                            image[row_index + x] = True

        # Extract pixels from image
        pixels = []
        for y in range(0, height, upscale):
            for x in range(0, width, upscale):
                opacity = 0
                for yy in range(0, upscale):
                    for xx in range(0, upscale):
                        if image[(y + yy) * width + (x + xx)]:
                            opacity = opacity + 255

                if opacity > 0:
                    pixels.append(
                        Pixel(
                            x=(x - shift_x) * downscale,
                            y=(y - shift_y) * downscale,
                            alpha=round(opacity * downscale ** 2),
                        )
                    )

        return pixels
