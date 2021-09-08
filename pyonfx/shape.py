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
"""Shape module"""
from __future__ import annotations

__all__ = ['Shape']

import re
from enum import Enum
from itertools import zip_longest
from math import atan, ceil, cos, degrees, inf, radians, sqrt
from typing import (Callable, Dict, Iterable, List, MutableSequence,
                    NamedTuple, NoReturn, Optional, Sequence, SupportsIndex,
                    Tuple, cast, overload)

import numpy as np
from more_itertools import unzip
from skimage.draw import polygon as skimage_polygon  # type: ignore
from skimage.transform import rescale as skimage_rescale  # type: ignore

from .colourspace import Opacity
from .geometry import (curve4_to_lines, get_line_intersect, get_ortho_vector,
                       get_vector, get_vector_angle, get_vector_length,
                       make_ellipse, make_parallelogram, make_triangle,
                       rotate_point, split_line, stretch_vector)
from .misc import chunk
from .types import Alignment, CoordinatesView, OutlineMode, PropView


class Pixel(NamedTuple):
    """A simple NamedTuple to represent pixels"""
    x: float
    y: float
    opacity: Opacity

    def to_ass_pixel(self, shift_x: float = 0, shift_y: float = 0,
                     round_digits: int = 3, optimise: bool = True) -> str:
        """
        Convenience function to get a ready-made line for the current pixel

        :param shift_x:         Shift number to add to the pixel abscissa, default to 0
        :param shift_y:         Shift number to add to the pixel ordinate, default to 0
        :param round_digits:    Decimal digits rounding precision, defaults to 3
        :param optimise:        Optimise the string by not adding the alpha tag when the pixel
                                is fully opaque, defaults to True
        :return:                Pixel in ASS format
        """
        if optimise:
            alpha = f'\\alpha{self.opacity}' if self.opacity.value < 1.0 else ''
        else:
            alpha = f'\\alpha{self.opacity}'
        return (
            f'{{\\p1\\pos({round(self.x + shift_x, round_digits)},{round(self.y + shift_y, round_digits)})'
            + alpha + f'}}{Shape.square(1).to_str()}'
        )


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

    def round(self, ndigits: int = 3, /) -> None:
        """
        Round coordinates to a given precision in decimal digits.

        :param ndigits:         Number of digits
        """
        for cmd in self:
            cmd.round(ndigits)

    def map(self, func: Callable[[float, float], Tuple[float, float]], /) -> None:
        """
        Sends every point of a shape through given transformation function to change them.

        **Tips:** *Working with outline points can be used to deform the whole shape and make f.e. a wobble effect.*

        :param func:            A function with two parameters representing the x and y coordinates of each point.
                                It will define how each coordinate will be changed.
        """
        for i, cmd in enumerate(self):
            self[i] = DrawingCommand(cmd.prop, *[func(x, y) for x, y in cmd.coordinates])

    def move(self, _x: float = 0., _y: float = 0., /) -> None:
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
            raise ValueError(f'{self.__class__.__name__}: Wrong "an" value!') from key_err
        x0, y0, x1, y1 = self.bounding()
        self.map(
            lambda x, y:
            (
                x + x0 * -1 + an_x * (x1 + x0 * -1),
                y + y0 * -1 + an_y * (y1 + y0 * -1)
            )
        )

    def rotate_x(self, rotation: float, zero_pad: Tuple[float, float] = (0., 0.), /) -> None:
        """
        Rotate current shape to a given rotation in the X-axis

        :param rotation:        Rotation in degrees
        :param zero_pad:        Point where the rotation will be performed, defaults to (0., 0.)
        """
        self.rotate((rotation, 0., 0.), zero_pad)

    def rotate_y(self, rotation: float, zero_pad: Tuple[float, float] = (0., 0.), /) -> None:
        """
        Rotate current shape to a given rotation in the Y-axis

        :param rotation:        Rotation in degrees
        :param zero_pad:        Point where the rotation will be performed, defaults to (0., 0.)
        """
        self.rotate((0., rotation, 0.), zero_pad)

    def rotate_z(self, rotation: float, zero_pad: Tuple[float, float] = (0., 0.), /) -> None:
        """
        Rotate current shape to a given rotation in the Z-axis

        :param rotation:        Rotation in degrees
        :param zero_pad:        Point where the rotation will be performed, defaults to (0., 0.)
        """
        self.map(lambda x, y: rotate_point(x, y, *zero_pad, rotation))

    def rotate(self, rotations: Tuple[float, float, float], zero_pad: Tuple[float, float] = (0., 0.), /,
               origin: Tuple[float, float] = (0., 0.), offset: Tuple[float, float] = (0., 0.)) -> None:
        """
        Rotate current shape to given rotations for all axis

        :param rotations:       Rotations in degrees in this order of X, Y and Z axis
        :param zero_pad:        Point where the rotations will be performed, defaults to (0., 0.)
        :param origin:          Origin anchor, defaults to (0., 0.)
        :param offset:          Offset coordinates, defaults to (0., 0.)
        """
        def _func(x: float, y: float) -> Tuple[float, float]:
            zpx, zpy = zero_pad
            x -= zpx
            y -= zpy
            x, y = rotate_and_project(x, y, 0., rotations, origin, offset)
            x += zpx
            y += zpy
            return x, y

        self.map(_func)

    def shear(self, fax: float = 0., fay: float = 0., /) -> None:
        """
        Perform a shearing (perspective distortion) transformation of the text.
        A factor of 0 means no distortion.

        :param fax:             X-axis factor, defaults to 0.
        :param fay:             Y-axis factor, defaults to 0.
        """
        self.map(
            lambda x, y:  # type: ignore
            tuple(map(float, np.array([(1, fax), (fay, 1)]) @ np.array([(x, ), (y, )])))
        )

    def close(self) -> None:
        """
        Close current shape if last point is not the same as the first one
        """
        if self[-1].coordinates != self[0].coordinates:
            self.append(DrawingCommand(DrawingProp.LINE, *tuple(self[0].coordinates)))

    def unclose(self) -> None:
        """
        Unclose current shape if last point is the same as the first one
        """
        if self[-1].coordinates == self[0].coordinates and self[-1].prop == DrawingProp.LINE:
            del self[-1]

    def split_shape(self) -> List[Shape]:
        """
        Split current shape into a list of Shape bounded
        by each DrawingProp.MOVE in the current shape object

        :return:                List of Shape objects
        """
        m_indx = [i for i, cmd in enumerate(self) if cmd.prop == DrawingProp.MOVE]
        return [self[i:j] for i, j in zip_longest(m_indx, m_indx[1:])]

    @classmethod
    def merge_shapes(cls, shapes: List[Shape]) -> Shape:
        """
        Merge the shapes into one Shape object

        :param shapes:          List of Shape objects
        :return:                A new merged Shape
        """
        return sum(shapes[1:], shapes[0])

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
                flatten_cmds = [DrawingCommand(l, c) for c in curve4_to_lines((cmd1[-1], *cmd0), tolerance)]  # type: ignore
                ncmds.extend(reversed(flatten_cmds))
            else:
                raise NotImplementedError(f'{self.__class__.__name__}: drawing property not recognised!')

        self.clear()
        self.extend(reversed(ncmds))

    def split_lines(self, max_length: float = 16., tolerance: float = 1.) -> None:
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
                splitted_cmds = [DrawingCommand(l, c) for c in split_line(cmd1[-1], cmd0[0], max_length)]
                ncmds.extend(reversed(splitted_cmds))
            else:
                raise NotImplementedError(f'{self.__class__.__name__}: drawing property not recognised!')

        self.clear()
        self.extend(reversed(ncmds))

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
        # Aliases
        DP = DrawingCommand
        m, b = DrawingProp.MOVE, DrawingProp.BÉZIER

        coordinates = make_ellipse(w, h, c_xy, clockwise)

        cmds = [
            DP(m, coordinates[0]),  # Start from bottom center
            DP(b, *coordinates[1]),
            DP(b, *coordinates[2]),
            DP(b, *coordinates[3]),
            DP(b, *coordinates[4]),
        ]
        return cls(cmds)

    @classmethod
    def heart(cls, size: float = 30., voffset: float = 0., /) -> Shape:
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
        m, l = DrawingProp.MOVE, DrawingProp.LINE

        coordinates = make_parallelogram(w, h, angle, c_xy, clockwise)

        cmds = [
            DC(m, coordinates[0]),
            DC(l, coordinates[1]),
            DC(l, coordinates[2]),
            DC(l, coordinates[3]),
            DC(l, coordinates[4])
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
                 c_xy: Tuple[float, float] = (0., 0.), /, clockwise: bool = True, *, orthocentred: bool = True) -> Shape:
        DC = DrawingCommand
        m, l = DrawingProp.MOVE, DrawingProp.LINE

        coordinates = make_triangle(side, angle, c_xy, clockwise)

        cmds = [
            DC(m, coordinates[0]),
            DC(l, coordinates[1]),
            DC(l, coordinates[2]),
            DC(l, coordinates[3]),
        ]

        triangle = cls(cmds)

        if orthocentred:
            _, yb0, _, yb1 = triangle.bounding()
            triangle.move(0, (yb1 - yb0) / 6)

        return triangle

    @classmethod
    def star(cls, edges: int, inner_size: float, outer_size: float, c_xy: Tuple[float, float] = (0., 0.)) -> Shape:
        """
        Make a star Shape object with given number of outer edges and sizes, centered around (c_xy).
        Use DrawingProp.LINE

        **Tips:** *Different numbers of edges and edge distances allow individual n-angles.*

        :param edges:           Number of edges
        :param inner_size:      Inner edges distance from center
        :param outer_size:      Outer edges distance from center
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :return:                A Shape object representing a star
        """
        return cls.stellation(edges, inner_size, outer_size, DrawingProp.LINE, c_xy)

    @classmethod
    def starfish(cls, edges: int, inner_size: float, outer_size: float, c_xy: Tuple[float, float] = (0., 0.)) -> Shape:
        """
        Make a starfish Shape object with given number of outer edges and sizes, centered around (c_xy).
        Use DrawingProp.CUBIC_BSPLINE

        :param edges:           Number of edges
        :param inner_size:      Inner edges distance from center
        :param outer_size:      Outer edges distance from center
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :return:                A Shape object representing a starfish
        """
        return cls.stellation(edges, inner_size, outer_size, DrawingProp.CUBIC_BSPLINE, c_xy)

    @classmethod
    def glance(cls, edges: int, inner_size: float, outer_size: float, c_xy: Tuple[float, float] = (0., 0.)) -> Shape:
        """
        Make a glance Shape object with given number of outer edges and sizes, centered around (c_xy).
        Use DrawingProp.BÉZIER

        **Tips:** *Glance is similar to star, but with curves instead of inner edges between the outer edges.*

        :param edges:           Number of edges
        :param inner_size:      Inner edges distance from center
        :param outer_size:      Control points for bezier curves between edges distance from center
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :return:                A Shape object representing a glance
        """
        return cls.stellation(edges, inner_size, outer_size, DrawingProp.BÉZIER, c_xy)

    @classmethod
    def stellation(cls, edges: int, inner_size: float, outer_size: float,
                   prop: DrawingProp, c_xy: Tuple[float, float] = (0., 0.)) -> Shape:
        """
        Make a stellationable Shape object with given number of outer edges and sizes, centered around (c_xy).
        Support DrawingProp.LINE, DrawingProp.CUBIC_BSPLINE and DrawingProp.BÉZIER.

        :param edges:           Number of edges
        :param inner_size:      Inner edges distance from center
        :param outer_size:      Outer edges distance from center
        :param prop:            Property used to build the shape
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :return:                A Shape object representing a stellationable shape
        """
        # https://en.wikipedia.org/wiki/Stellation
        DC = DrawingCommand
        m, l, b, s = DrawingProp.MOVE, DrawingProp.LINE, DrawingProp.BÉZIER, DrawingProp.BSPLINE

        cmds: List[DrawingCommand] = []
        coordinates: List[Tuple[float, float]] = []

        cmds.append(DC(m, (0, -outer_size)))

        for i in range(1, edges + 1):
            inner_p = rotate_point(0, -inner_size, 0, 0, ((i - 0.5) / edges) * 360)
            outer_p = rotate_point(0, -outer_size, 0, 0, (i / edges) * 360)
            if prop == l:
                cmds += [DC(prop, inner_p), DC(prop, outer_p)]
            elif prop == b:
                cmds += [DC(prop, inner_p, inner_p, outer_p)]
            elif prop == s:
                coordinates += [inner_p, inner_p, outer_p]
                if i == edges:
                    cmds += [DC(DrawingProp.BSPLINE, *coordinates[:-1])] + [DC(DrawingProp.CLOSE_BSPLINE)]
            else:
                raise NotImplementedError(f'{cls.__name__}: prop "{prop}" not supported!')

        shape = cls(cmds)
        shape.move(*c_xy)

        return shape

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

    def to_pixels(self, supersampling: int = 4, anti_aliasing: bool = True) -> List[Pixel]:
        """
        Convert current Shape to a list of Pixel
        It is strongly recommended to create a dedicated style for pixels,
        thus, you will write less tags for line in your pixels,
        which means less size for your .ass file.

        Style suggested as an=7, bord=0, shad=0

        :param supersampling:       Supersampling value to avoid aliasing.
                                    Higher value means smoother and more precise anti-aliasing
                                    (and more computational time for generation), defaults to 4
        :param anti_aliasing:       Downscale with anti_aliasing or not, default to True
        :return:                    List of Pixel
        """
        # Copy current shape object
        wshape = Shape(self)
        # Get shift
        shift_x, shift_y, _, _ = wshape.bounding()
        # Align shape to 7 - Positive coordinates
        wshape.align(7)
        # Upscale it
        wshape.map(lambda x, y: (x * supersampling, y * supersampling))

        # Flatten the shape to avoid working with bézier curves
        wshape.flatten()

        def _close_shape(shape: Shape) -> Shape:
            shape.close()
            return shape

        # Close each shape and merge again them
        wshape = self.merge_shapes([_close_shape(wsh) for wsh in wshape.split_shape()])

        # Build an image
        _, _, x1, y1 = wshape.bounding()
        width, height = ceil(x1 + supersampling * 2), ceil(y1 + supersampling * 2)
        image = np.zeros((height, width), np.uint8)

        # Extract coordinates
        xs, ys = unzip(c for cv in wshape.coordinates for c in cv)
        # Build rows and columns from coordinates
        rows, columns = np.array(list(ys)), np.array(list(xs))
        # Get polygons coordinates
        rr, cc = skimage_polygon(rows, columns)
        # Fill the image from the polygon coordinates
        image[rr, cc] = 255
        # Downscale while avoiding aliasing
        image = skimage_rescale(
            image, 1 / supersampling,
            preserve_range=True, anti_aliasing=anti_aliasing
        )
        # Return all those pixels
        return [
            Pixel(int(x - shift_x), int(y - shift_y), Opacity(alpha / 255))
            for y, row in enumerate(image)
            for x, alpha in enumerate(row)
            if alpha > 0
        ]

    def to_outline(self, bord_xy: float, bord_y: Optional[float] = None, mode: OutlineMode = 'round', *,
                   miter_limit: float = 200., max_circumference: float = 2.) -> None:
        """
        Converts Shape command for filling to a Shape command for stroking.

        :param bord_xy:             Outline border
        :param bord_y:              Y-axis outline border, defaults to None
        :param mode:                Stroking mode, can be 'miter', 'bevel' ou 'round', defaults to "round"
        """
        if len(self) < 2:
            raise ValueError(f'{self.__class__.__name__}: Shape must have at least 2 commands')

        # -- Line width values
        if bord_y and bord_xy != bord_y:
            width = max(bord_xy, bord_y)
            xscale, yscale = bord_xy / width, bord_y / width
        else:
            width, xscale, yscale = bord_xy, 1, 1

        self.flatten()
        shapes = self.split_shape()

        def _unclose_shape(sh: Shape) -> Shape:
            sh.unclose()
            return sh

        shapes = [_unclose_shape(s) for s in shapes]

        # -- Create stroke shape out of figures
        DC, DP = DrawingCommand, DrawingProp
        m, l = DP.MOVE, DP.LINE
        stroke_cmds: List[DrawingCommand] = []

        for shape in shapes:
            # Outer
            rcmds = [shape[0]] + list(reversed(shape[1:]))
            outline = self._stroke_lines(rcmds, width, xscale, yscale, mode, miter_limit, max_circumference)
            stroke_cmds.append(DC(m, outline.pop(0)))
            stroke_cmds.extend(DC(l, coordinate) for coordinate in outline)

            # Inner
            outline = self._stroke_lines(shape, width, xscale, yscale, mode, miter_limit, max_circumference)
            stroke_cmds.append(DC(m, outline.pop(0)))
            stroke_cmds.extend(DC(l, coordinate) for coordinate in outline)

        self.clear()
        self.extend(stroke_cmds)

    def _stroke_lines(self, shape: MutableSequence[DrawingCommand], width: float,
                      xscale: float, yscale: float, mode: str,
                      miter_limit: float, max_circumference: float) -> List[Tuple[float, float]]:
        outline: List[Tuple[float, float]] = []
        for point, pre_point, post_point in zip(
            shape,
            [shape[-1]] + list(shape[:-1]),
            list(shape[1:]) + [shape[0]]
        ):
            # -- Calculate orthogonal vectors to both neighbour points
            p, pre_p, post_p = point[0], pre_point[0], post_point[0]
            vec0, vec1 = get_vector(p, pre_p), get_vector(p, post_p)

            o_vec0 = get_ortho_vector(*((*vec0, 0.), (0., 0., 1.)))
            o_vec0 = stretch_vector(o_vec0[:-1], width)

            o_vec1 = get_ortho_vector(*((*vec1, 0.), (0., 0., -1.)))
            o_vec1 = stretch_vector(o_vec1[:-1], width)

            # -- Check for gap or edge join
            inter_x, inter_y = get_line_intersect(
                (p[0] + o_vec0[0] - vec0[0], p[1] + o_vec0[1] - vec0[1]),
                (p[0] + o_vec0[0],           p[1] + o_vec0[1]),  # noqa: E241
                (p[0] + o_vec1[0] - vec1[0], p[1] + o_vec1[1] - vec1[1]),
                (p[0] + o_vec1[0],           p[1] + o_vec1[1]),  # noqa: E241
                True
            )
            if inter_y != inf:
                # -- Add gap point
                outline.append(
                    (p[0] + (inter_x - p[0]) * xscale, p[1] + (inter_y - p[1]) * yscale)
                )
            else:
                # -- Add first edge point
                outline.append(
                    (p[0] + o_vec0[0] * xscale, p[1] + o_vec0[1] * yscale)
                )
                # -- Create join by mode
                if mode == 'bevel':
                    continue
                if mode == 'miter':
                    outline.extend(self._join_mode_miter(p, vec0, vec1, o_vec0, o_vec1, xscale, yscale, miter_limit))

                elif mode == 'round':
                    outline.extend(self._join_mode_round(p, o_vec0, o_vec1, xscale, yscale, width, max_circumference))
                else:
                    raise ValueError(f'"{mode}" is not a supported mode!')
                # -- Add end edge point
                outline.append(
                    (p[0] + o_vec1[0] * xscale, p[1] + o_vec1[1] * yscale)
                )
        return outline

    @staticmethod
    def _join_mode_miter(
        p: Tuple[float, float],
        vec0: Tuple[float, float], vec1: Tuple[float, float],
        o_vec0: Tuple[float, float], o_vec1: Tuple[float, float],
        xscale: float, yscale: float, miter_limit: float
    ) -> List[Tuple[float, float]]:
        """Internal function"""
        outline: List[Tuple[float, float]] = []

        inter_x, inter_y = get_line_intersect(
            (p[0] + o_vec0[0] - vec0[0], p[1] + o_vec0[1] - vec0[1]),
            (p[0] + o_vec0[0],           p[1] + o_vec0[1]),  # noqa: E241
            (p[0] + o_vec1[0] - vec1[0], p[1] + o_vec1[1] - vec1[1]),
            (p[0] + o_vec1[0],           p[1] + o_vec1[1]),  # noqa: E241
            strict=False
        )
        # -- Vectors intersect
        if inter_y != inf:
            is_vec_x, is_vec_y = inter_x - p[0], inter_y - p[1]
            is_vec_len = get_vector_length((is_vec_x, is_vec_y))
            if is_vec_len > miter_limit:
                fix_scale = miter_limit / is_vec_len
                outline.append(
                    (
                        p[0] + (o_vec0[0] + (is_vec_x - o_vec0[0]) * fix_scale) * xscale,
                        p[1] + (o_vec0[1] + (is_vec_y - o_vec0[1]) * fix_scale) * yscale
                    )
                )
                outline.append(
                    (
                        p[0] + (o_vec1[0] + (is_vec_x - o_vec1[0]) * fix_scale) * xscale,
                        p[1] + (o_vec1[1] + (is_vec_y - o_vec1[1]) * fix_scale) * yscale
                    )
                )
            else:
                outline.append(
                    (p[0] + is_vec_x * xscale, p[1] + is_vec_y * yscale)
                )
        # -- Parallel vectors
        else:
            vec0, vec1 = stretch_vector(vec0, miter_limit), stretch_vector(vec1, miter_limit)
            outline.append(
                (p[0] + (o_vec0[0] + vec0[0]) * xscale, p[1] + (o_vec0[1] + vec0[1]) * yscale)
            )
            outline.append(
                (p[0] + (o_vec1[0] + vec1[0]) * xscale, p[1] + (o_vec1[1] + vec1[1]) * yscale)
            )
        return outline

    @staticmethod
    def _join_mode_round(
        p: Tuple[float, float],
        o_vec0: Tuple[float, float], o_vec1: Tuple[float, float],
        xscale: float, yscale: float, width: float, max_circumference: float
    ) -> List[Tuple[float, float]]:
        """Internal function"""
        outline: List[Tuple[float, float]] = []

        # -- Calculate degree & circumference between orthogonal vectors
        degree = get_vector_angle(o_vec0, o_vec1)
        circ = abs(radians(degree)) * width

        if circ > max_circumference:
            # -- Add curve edge points
            circ_rest = circ % max_circumference
            for cur_circ in np.arange(
                circ_rest if circ_rest > 0 else max_circumference,
                circ, max_circumference
            ):
                curve_vec = rotate_point(*o_vec0, 0, 0, cur_circ / circ * degree)
                outline.append(
                    (p[0] + curve_vec[0] * xscale, p[1] + curve_vec[1] * yscale)
                )
        return outline
