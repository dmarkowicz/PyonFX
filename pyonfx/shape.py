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

__all__ = [
    'Shape',
    'Pixel',
    'DrawingProp',
    'DrawingCommand',
    'OutlineMode'
]

import re
from enum import Enum, auto
from itertools import zip_longest
from math import atan, ceil, cos, degrees, inf, radians, sqrt
from typing import (
    Callable, Dict, Iterable, List, MutableSequence, NamedTuple, NoReturn, Optional, Sequence,
    SupportsIndex, Tuple, cast, overload
)

import numpy as np
from more_itertools import flatten, sliced, unzip, zip_offset
from skimage.draw import polygon as skimage_polygon  # type: ignore
from skimage.transform import rescale as skimage_rescale  # type: ignore

from .colourspace import ASSColor, Opacity
from .geometry import CartesianAxis, Geometry, Point, PointCartesian2D, PointsView, VectorCartesian2D, VectorCartesian3D
from .misc import chunk, clamp_value, frange
from .types import Alignment, View


class Pixel(NamedTuple):
    """A simple NamedTuple to represent pixels"""
    pos: PointCartesian2D
    opacity: Optional[Opacity] = None
    colour: Optional[ASSColor] = None

    def to_ass_pixel(self, shift_x: float = 0, shift_y: float = 0, round_digits: int = 3) -> str:
        """
        Convenience function to get a ready-made line for the current pixel

        :param shift_x:         Shift number to add to the pixel abscissa, default to 0
        :param shift_y:         Shift number to add to the pixel ordinate, default to 0
        :param round_digits:    Decimal digits rounding precision, defaults to 3
        :return:                Pixel in ASS format
        """
        alpha = (
            f'\\alpha{self.opacity}' if self.opacity.data not in {'&HFF&', '&H00&'} else ''
        ) if self.opacity is not None else ''
        colour = f'\\c{self.colour}' if self.colour is not None else ''
        return (
            f'{{\\p1\\pos({round(self.pos.x + shift_x, round_digits)},{round(self.pos.y + shift_y, round_digits)})'
            + alpha + colour + f'}}{Shape.square(1.5).to_str()}'
        )


class OutlineMode(Enum):
    """Simple enum class for OutlineMode in Shape.to_outline method"""
    MITER = auto()
    BEVEL = auto()
    ROUND = auto()


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


class PropsView(View[DrawingProp]):
    """View for DrawingProps"""
    ...


class DrawingCommand(Sequence[Point]):
    """
    A drawing command is a DrawingProp and a number of coordinates
    """
    _prop: DrawingProp
    _coordinates: List[Point]

    @property
    def prop(self) -> DrawingProp:
        """The DrawingProp of this DrawingCommand"""
        return self._prop

    @property
    def coordinates(self) -> PointsView:
        """Coordinates of this DrawingCommand"""
        return PointsView(self._coordinates)

    def __init__(self, prop: DrawingProp, *coordinates: Tuple[float, float] | Point) -> None:
        """
        Make a DrawingCommand object

        :param prop:            Drawing property of this DrawingCommand
        :param coordinates:     Coordinates of this DrawingCommand
        """
        self._prop = prop
        self._coordinates = [c if isinstance(c, Point) else PointCartesian2D(*c) for c in coordinates]
        self.check_integrity()
        super().__init__()

    @overload
    def __getitem__(self, index: SupportsIndex) -> Point:
        ...

    @overload
    def __getitem__(self, index: slice) -> NoReturn:
        ...

    def __getitem__(self, index: SupportsIndex | slice) -> Point | NoReturn:
        if isinstance(index, SupportsIndex):
            return self._coordinates[index]
        raise NotImplementedError(f'{self.__class__.__name__}: slice is not supported!')

    def __len__(self) -> int:
        return len(self._coordinates)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DrawingCommand):
            return NotImplemented
        return o._prop == self._prop and o._coordinates == self._coordinates

    def __str__(self) -> str:
        return self._prop.value + ' ' + ' '.join(
            f'{p.x} {p.y}'
            for p in [
                # Get the points and convert them to 2D
                po if isinstance(po, PointCartesian2D) else po.to_3d().project_2d() for po in self
            ]
        )

    def to_str(self, round_digits: int = 3, optimise: bool = True) -> str:
        """
        Return the current command in ASS format

        :param round_digits:    Decimal digits rounding precision, defaults to 3
        :param optimise:        Optimise the string by removing redundant drawing prop, defaults to True
        :return:                Shape in ASS format
        """
        if optimise:
            points: List[PointCartesian2D] = []
            for po in self:
                # Get the points and convert them to 2D
                po = po if isinstance(po, PointCartesian2D) else po.to_3d().project_2d()
                po.round(round_digits)
                # Optimise by removing ".0" if the float can be interpreted as an integer
                if (intx := int(po.x)) == po.x:
                    po.x = intx
                if (inty := int(po.y)) == po.y:
                    po.y = inty
                points.append(po)
            return self._prop.value + ' ' + ' '.join(f'{p.x} {p.y}' for p in points)
        return str(self)

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
        for p in self:
            p.round(ndigits)


class Shape(MutableSequence[DrawingCommand]):
    """
    Class for creating, handling, making transformations from an ASS shape
    """
    _commands: List[DrawingCommand]

    @property
    def props(self) -> PropsView:
        """The DrawingProp of this DrawingCommand"""
        return PropsView([c.prop for c in self])

    @property
    def coordinates(self) -> List[PointsView]:
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

    def replace(self, other: Iterable[DrawingCommand]) -> None:
        """
        Replace every DrawingCommand in the current Shape by other's DrawingCommand

        :param other:           Iterable of DrawingCommands
        """
        self.clear()
        self.extend(other)

    def to_str(self, round_digits: int = 3, optimise: bool = True) -> str:
        """
        Return the current shape in ASS format

        :param round_digits:    Decimal digits rounding precision, defaults to 3
        :param optimise:        Optimise the string by removing redundant drawing prop, defaults to True
        :return:                Shape in ASS format
        """
        self.round(round_digits)

        if optimise:
            # Last prop used, drawing to be str'd
            p, draw = DrawingProp.CLOSE_BSPLINE, ''

            # Iterating over all the commands
            for cmd in self:
                cmdstr = cmd.to_str(round_digits)
                if cmd.prop != p:
                    draw += cmdstr + ' '
                elif cmd.prop == p and cmd.prop in {DrawingProp.LINE, DrawingProp.CUBIC_BÉZIER_CURVE}:
                    draw += cmdstr[2:] + ' '
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

    def map(self, func: Callable[[Point], Point | Tuple[float, float]], /) -> None:
        """
        Sends every point of a shape through given transformation function to change them.

        **Tips:** *Working with outline points can be used to deform the whole shape and make f.e. a wobble effect.*

        :param func:            A function with two parameters representing the x and y coordinates of each point.
                                It will define how each coordinate will be changed.
        """
        for i, cmd in enumerate(self):
            self[i] = DrawingCommand(cmd.prop, *[func(p) for p in cmd.coordinates])

    def move(self, _x: float = 0., _y: float = 0., /) -> None:
        """
        Moves shape coordinates in given direction.

        :param _x:              Displacement along the x-axis, defaults to 0.
        :param _y:              Displacement along the y-axis, defaults to 0.
        """
        def _func(p: Point) -> Point:
            p = p.to_2d()
            p.x += _x
            p.y += _y
            return p
        self.map(_func)

    def scale(self, _x: float = 1., _y: float = 1., /) -> None:
        """
        Scale shape coordinates by given factors

        :param _x:              X-axis scale factor, defaults to 1.
        :param _y:              Y-axis scale factor, defaults to 1.
        """
        def _func(p: Point) -> Point:
            p = p.to_2d()
            p.x *= _x
            p.y *= _y
            return p
        self.map(_func)

    @property
    def bounding(self) -> Tuple[PointCartesian2D, PointCartesian2D]:
        """
        Calculates shape bounding box.

        **Tips:** *Using this you can get more precise information about a shape (width, height, position).*

        Examples:
            ..  code-block:: python3

                x0, y0, x1, y1 = Shape.from_ass_string("m 10 5 l 25 5 25 42 10 42").bounding
                print(f"Left-top: {x0} {y0}\\nRight-bottom: {x1} {y1}")

            >>> Left-top: 10 5
            >>> Right-bottom: 25 42

        :return:                A tuple of coordinates of the bounding box
        """
        all_x, all_y = [
            set(c) for c in unzip([c.to_2d() for dc in self for c in dc])
        ]

        return PointCartesian2D(min(all_x), min(all_y)), PointCartesian2D(max(all_x), max(all_y))

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
        pb0, pb1 = self.bounding
        self.move(
            pb0.x * -1 + an_x * (pb1.x + pb0.x * -1),
            pb0.y * -1 + an_y * (pb1.y + pb0.y * -1)
        )

    def rotate(self, rot: float, axis: CartesianAxis, /, zero_pad: Optional[Tuple[float, ...]] = (0., 0., 0.)) -> None:
        """
        Rotate current shape to given rotation in given axis

        :param rot:             Rotation in degrees
        :param axis:            Axis where the rotation will be performed
        :param zero_pad:        Point where the rotations will be performed, defaults to None
        """
        self.map(lambda p: Geometry.rotate(p.to_3d(), rot, axis, zero_pad).project_2d())

    def shear(self, fax: float = 0., fay: float = 0., /) -> None:
        """
        Perform a shearing (perspective distortion) transformation of the text.
        A factor of 0 means no distortion.

        :param fax:             X-axis factor, defaults to 0.
        :param fay:             Y-axis factor, defaults to 0.
        """
        def _func(p: Point) -> PointCartesian2D:
            p = p.to_2d()
            return PointCartesian2D(*map(float, np.array([(1, fax), (fay, 1)]) @ p))

        self.map(_func)

    def close(self) -> None:
        """
        Close current shape if last point is not the same as the first one
        """
        if self.coordinates[0] != self.coordinates[-1]:
            self.append(DrawingCommand(DrawingProp.LINE, next(reversed(self.coordinates[0]))))

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
        m_indx = [i for i, cmd in enumerate(self) if cmd.prop in {DrawingProp.MOVE, DrawingProp.MOVE_NC}]
        return [self[i:j] for i, j in zip_offset(m_indx, m_indx, offsets=(0, 1), longest=True)]

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

        for cmd0, cmd1 in zip_offset(self, self, offsets=(0, 1), longest=True, fillvalue=DrawingCommand(m, (0, 0))):
            if cmd0.prop in {m, n, l}:
                ncmds.append(cmd0)
            elif cmd0.prop in {p, s, c}:
                raise NotImplementedError(
                    f'{self.__class__.__name__}: EXTEND_BSPLINE, CUBIC_BSPLINE and CLOSE_BSPLINE'
                    + ' drawing properties are not supported!'
                )
            elif cmd0.prop == b:
                # Get the previous coordinate to complete a bezier curve
                flatten_cmds = [DrawingCommand(l, co) for co in Geometry.curve4_to_lines((cmd1[-1], *cmd0), tolerance)]  # type: ignore
                ncmds.extend(reversed(flatten_cmds))
            else:
                raise NotImplementedError(f'{self.__class__.__name__}: drawing property not recognised!')

        self.replace(reversed(ncmds))

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
                splitted_cmds = [
                    DrawingCommand(l, c) for c in Geometry.split_line(cmd1[-1].to_2d(), cmd0[0].to_2d(), max_length)
                ]
                ncmds.extend(reversed(splitted_cmds))
            else:
                raise NotImplementedError(f'{self.__class__.__name__}: drawing property not recognised!')

        self.replace(reversed(ncmds))

    def round_vertices(self, deviation: float = 15, tolerance: float = 157.5, tension: float = 0.5) -> None:
        """
        Round vertices of the current shape

        :param deviation:       Length in pixel of the deviation from each vertex, defaults to 15
        :param tolerance:       Angle in degree to define a vertex to be rounded.
                                If the vertex's angle is lower than tolerance then it will be rounded.
                                Valid ranges are 0.0 - 180.0, defaults to 157.5
        :param tension:         Adjust point tension in percentage, defaults to 0.5
        """
        # Aliases
        DP = DrawingProp
        m, n, l, b = DP.MOVE, DP.MOVE_NC, DP.LINE, DP.BÉZIER

        shapes = self.split_shape()

        for shape in shapes:
            shape.unclose()
            ncmds: List[DrawingCommand] = []
            ncmds.clear()

            pres = list(shape)
            pres.insert(0, pres.pop(-1))
            posts = list(shape)
            posts.append(posts.pop(0))

            for pre, curr, post in zip(pres, shape, posts):
                if curr.prop in {m, n, l}:
                    curve = Geometry.round_vertex(
                        pre[-1].to_2d(), curr[0].to_2d(), post[0].to_2d(),
                        deviation, tolerance, clamp_value(tension, 0., 1.)
                    )
                    ncmds.append(DrawingCommand(curr.prop, curve.pop(0)))
                    if curve:
                        ncmds.append(DrawingCommand(b, *curve))
                else:
                    ncmds.append(curr)
            shape.replace(ncmds)
        self.replace(flatten(shapes))

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

        coordinates = Geometry.make_ellipse(w, h, c_xy, clockwise)

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

        coordinates = Geometry.make_parallelogram(w, h, angle, c_xy, clockwise)

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

        coordinates = Geometry.make_triangle(side, angle, c_xy, clockwise)

        cmds = [
            DC(m, coordinates[0]),
            DC(l, coordinates[1]),
            DC(l, coordinates[2]),
            DC(l, coordinates[3]),
        ]

        triangle = cls(cmds)

        if orthocentred:
            pb0, pb1 = triangle.bounding
            triangle.move(0, (pb1.y - pb0.y) / 6)

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
        coordinates: List[PointCartesian2D] = []

        cmds.append(DC(m, (0, -outer_size)))

        for i in range(1, edges + 1):
            inner_p = Geometry.rotate(PointCartesian2D(0, -inner_size), ((i - 0.5) / edges) * 360, None, (0., 0.))
            outer_p = Geometry.rotate(PointCartesian2D(0, -outer_size), (i / edges) * 360, None, (0., 0.))
            if prop == l:
                cmds += [DC(prop, inner_p), DC(prop, outer_p)]
            elif prop == b:
                cmds += [DC(prop, inner_p, inner_p, outer_p)]
            elif prop == s:
                coordinates += [inner_p, inner_p, outer_p]
                if i == edges:
                    cmds += [DC(s, *coordinates[:-1])] + [DC(DrawingProp.CLOSE_BSPLINE)]
            else:
                raise NotImplementedError(f'{cls.__name__}: prop "{prop}" not supported!')

        shape = cls(cmds)
        if c_xy != (0., 0.):
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
                    ) for x, y in sliced(sdraw, 2, strict=True)
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
        ss = supersampling
        # Copy current shape object
        wshape = Shape(self)
        # Get shift
        shiftp, _ = wshape.bounding
        # Align shape to 7 - Positive coordinates
        wshape.align(7)
        # Upscale it
        wshape.scale(ss, ss)

        # Flatten the shape to avoid working with bézier curves
        wshape.flatten()

        def _close_shape(shape: Shape) -> Shape:
            shape.close()
            return shape

        # Close each shape and merge again them
        wshape = self.merge_shapes([_close_shape(wsh) for wsh in wshape.split_shape()])

        # Build an image
        _, pb1 = wshape.bounding
        width, height = ceil(pb1.x), ceil(pb1.y)
        image = np.zeros((height, width), np.uint8)  # type: ignore[var-annotated]

        # Extract coordinates
        xs, ys = unzip(c.to_2d() for cv in wshape.coordinates for c in cv)
        # Build rows and columns from coordinates
        rows, columns = np.array(list(ys)), np.array(list(xs))  # type: ignore[var-annotated]
        # Get polygons coordinates
        rr, cc = skimage_polygon(rows, columns, shape=(height, width))
        # Fill the image from the polygon coordinates
        image[rr, cc] = 255
        # Downscale while avoiding aliasing
        image = skimage_rescale(
            image, 1 / ss,
            preserve_range=True, anti_aliasing=anti_aliasing
        )
        # Return all those pixels
        return [
            Pixel(PointCartesian2D(x - shiftp.x, y - shiftp.y), Opacity(alpha / 255))
            for y, row in enumerate(image)
            for x, alpha in enumerate(row)
            if alpha > 0
        ]

    def to_outline(self, bord_xy: float, bord_y: Optional[float] = None,
                   mode: OutlineMode = OutlineMode.ROUND, *,
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

        self.replace(stroke_cmds)

    def _stroke_lines(self, shape: MutableSequence[DrawingCommand], width: float,
                      xscale: float, yscale: float, mode: OutlineMode,
                      miter_limit: float, max_circumference: float) -> List[PointCartesian2D]:
        outline: List[PointCartesian2D] = []
        for point, pre_point, post_point in zip(
            shape,
            [shape[-1]] + list(shape[:-1]),
            list(shape[1:]) + [shape[0]]
        ):
            # -- Calculate orthogonal vectors to both neighbour points
            p, pre_p, post_p = point[0].to_2d(), pre_point[0].to_2d(), post_point[0].to_2d()
            vec0, vec1 = Geometry.vector(p, pre_p), Geometry.vector(p, post_p)

            o_vec0 = Geometry.orthogonal(vec0.to_3d(), VectorCartesian3D(0., 0., 1.)).to_2d()
            o_vec0 = Geometry.stretch(o_vec0, width)

            o_vec1 = Geometry.orthogonal(vec1.to_3d(), VectorCartesian3D(0., 0., -1.)).to_2d()
            o_vec1 = Geometry.stretch(o_vec1, width)

            # -- Check for gap or edge join
            inter = Geometry.line_intersect(
                PointCartesian2D(p.x + o_vec0.x - vec0.x, p.y + o_vec0.y - vec0.y),
                PointCartesian2D(p.x + o_vec0.x,           p.y + o_vec0.y),  # noqa: E241
                PointCartesian2D(p.x + o_vec1.x - vec1.x, p.y + o_vec1.y - vec1.y),
                PointCartesian2D(p.x + o_vec1.x,           p.y + o_vec1.y),  # noqa: E241
                True
            )
            if inter.y != inf:
                # -- Add gap point
                outline.append(
                    PointCartesian2D(p.x + (inter.x - p.x) * xscale, p.y + (inter.y - p.y) * yscale)
                )
            else:
                # -- Add first edge point
                outline.append(
                    PointCartesian2D(p.x + o_vec0.x * xscale, p.y + o_vec0.y * yscale)
                )
                # -- Create join by mode
                if mode == OutlineMode.ROUND:
                    outline.extend(self._join_mode_round(p, o_vec0, o_vec1, xscale, yscale, width, max_circumference))
                elif mode == OutlineMode.MITER:
                    outline.extend(self._join_mode_miter(p, vec0, vec1, o_vec0, o_vec1, xscale, yscale, miter_limit))
                else:  # OutlineMode.BEVEL:
                    continue
                # -- Add end edge point
                outline.append(
                    PointCartesian2D(p.x + o_vec1.x * xscale, p.y + o_vec1.y * yscale)
                )
        return outline

    @staticmethod
    def _join_mode_miter(
        p: PointCartesian2D,
        vec0: VectorCartesian2D, vec1: VectorCartesian2D,
        o_vec0: VectorCartesian2D, o_vec1: VectorCartesian2D,
        xscale: float, yscale: float, miter_limit: float
    ) -> List[PointCartesian2D]:
        """Internal function"""
        outline: List[PointCartesian2D] = []

        inter = Geometry.line_intersect(
            PointCartesian2D(p.x + o_vec0.x - vec0.x, p.y + o_vec0.y - vec0.y),
            PointCartesian2D(p.x + o_vec0.x,           p.y + o_vec0.y),  # noqa: E241
            PointCartesian2D(p.x + o_vec1.x - vec1.x, p.y + o_vec1.y - vec1.y),
            PointCartesian2D(p.x + o_vec1.x,           p.y + o_vec1.y),  # noqa: E241
            strict=False
        )
        # -- Vectors intersect
        if inter.y != inf:
            is_vec = Geometry.vector(inter, p)
            is_vec_len = is_vec.norm
            if is_vec_len > miter_limit:
                fix_scale = miter_limit / is_vec_len
                outline.append(
                    PointCartesian2D(
                        p.x + (o_vec0.x + (is_vec.x - o_vec0.x) * fix_scale) * xscale,
                        p.y + (o_vec0.y + (is_vec.y - o_vec0.y) * fix_scale) * yscale
                    )
                )
                outline.append(
                    PointCartesian2D(
                        p.x + (o_vec1.x + (is_vec.x - o_vec1.x) * fix_scale) * xscale,
                        p.y + (o_vec1.y + (is_vec.y - o_vec1.y) * fix_scale) * yscale
                    )
                )
            else:
                outline.append(
                    PointCartesian2D(p.x + is_vec.x * xscale, p.y + is_vec.y * yscale)
                )
        # -- Parallel vectors
        else:
            vec0, vec1 = Geometry.stretch(vec0, miter_limit), Geometry.stretch(vec1, miter_limit)
            outline.append(
                PointCartesian2D(p.x + (o_vec0.x + vec0.x) * xscale, p.y + (o_vec0.y + vec0.y) * yscale)
            )
            outline.append(
                PointCartesian2D(p.x + (o_vec1.x + vec1.x) * xscale, p.y + (o_vec1.y + vec1.y) * yscale)
            )
        return outline

    @staticmethod
    def _join_mode_round(
        p: PointCartesian2D,
        o_vec0: VectorCartesian2D, o_vec1: VectorCartesian2D,
        xscale: float, yscale: float, width: float, max_circumference: float
    ) -> List[PointCartesian2D]:
        """Internal function"""
        outline: List[PointCartesian2D] = []

        # -- Calculate degree & circumference between orthogonal vectors
        degree = Geometry.angle(o_vec0, o_vec1)
        circ = abs(degree) * width

        if circ > max_circumference:
            # -- Add curve edge points
            circ_rest = circ % max_circumference
            for cur_circ in frange(
                circ_rest if circ_rest > 0 else max_circumference,
                circ, max_circumference
            ):
                curve_vec = Geometry.rotate(o_vec0, cur_circ / circ * degree, None, (0., 0.))
                outline.append(
                    PointCartesian2D(p.x + curve_vec.x * xscale, p.y + curve_vec.y * yscale)
                )
        return outline
