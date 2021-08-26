# -*- coding: utf-8 -*-
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

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Any, List, NamedTuple, NoReturn, Optional

from .colourspace import Opacity
from .font_utility import Font
from .types import Alignment

if TYPE_CHECKING:
    from .ass_core import AssText
    from .shape import Shape


class Pixel(NamedTuple):
    """A simple NamedTuple to represent pixels"""
    x: float
    y: float
    alpha: Opacity


class Convert:
    """
    This class is a collection of methods that will help
    the user to convert everything needed to the ASS format.
    """

    # Seconds | Timestamp
    @staticmethod
    def ts2seconds(ts: str, /) -> float:
        h, m, s = map(float, ts.split(':'))
        return h * 3600 + m * 60 + s

    @classmethod
    def seconds2ts(cls, s: float, /, *, precision: int = 3) -> str:
        m = s // 60
        s %= 60
        h = m // 60
        m %= 60
        return cls.composets(h, m, s, precision=precision)

    # Seconds | Frame
    @staticmethod
    def seconds2f(s: float, fps: Fraction, /) -> int:
        return round(s * fps)

    @staticmethod
    def f2seconds(f: int, fps: Fraction, /) -> float:
        if f == 0:
            return 0.0

        t = round(float(10 ** 9 * f * fps ** -1))
        s = t / 10 ** 9
        return s

    # Frame | Timestamp
    @classmethod
    def f2ts(cls, f: int, fps: Fraction, /, *, precision: int = 3) -> str:
        s = cls.f2seconds(f, fps)
        ts = cls.seconds2ts(s, precision=precision)
        return ts

    @classmethod
    def ts2f(cls, ts: str, fps: Fraction, /) -> int:
        s = cls.ts2seconds(ts)
        f = cls.seconds2f(s, fps)
        return f

    # Ass Timestamp | Seconds
    @classmethod
    def seconds2assts(cls, s: float, fps: Fraction, /) -> str:
        s -= fps ** -1 * 0.5
        ts = cls.seconds2ts(max(0, s), precision=3)
        return ts[:-1]

    @classmethod
    def assts2seconds(cls, assts: str, fps: Fraction, /) -> float:
        s = cls.ts2seconds(assts)
        if s > 0:
            s += fps ** -1 * 0.5
        return s

    @staticmethod
    def composets(h: float, m: float, s: float, /, *, precision: int = 3) -> str:
        if precision == 0:
            out = f"{h:02.0f}:{m:02.0f}:{round(s):02}"
        elif precision == 3:
            out = f"{h:02.0f}:{m:02.0f}:{s:06.3f}"
        elif precision == 6:
            out = f"{h:02.0f}:{m:02.0f}:{s:09.6f}"
        elif precision == 9:
            out = f"{h:02.0f}:{m:02.0f}:{s:012.9f}"
        else:
            raise ValueError(f'composets: the precision {precision} must be a multiple of 3 (including 0)')
        return out

    @classmethod
    def bound_to_frame(cls, s: float, fps: Fraction, /) -> float:
        return cls.f2seconds(cls.seconds2f(s, fps), fps)

    @staticmethod
    def text_to_shape(ass_text: AssText, fscx: Optional[float] = None, fscy: Optional[float] = None) -> Shape:
        """Converts text with given style information to an ASS shape.

        **Tips:** *You can easily create impressive deforming effects.*

        Parameters:
            obj (Line, Word, Syllable or Char): An object of class Line, Word, Syllable or Char.
            fscx (float, optional): The scale_x value for the shape.
            fscy (float, optional): The scale_y value for the shape.

        Returns:
            A Shape object, representing the text with the style format values of the object.

        Examples:
            ..  code-block:: python3

                line = Line.copy(lines[1])
                line.text = "{\\\\an7\\\\pos(%.3f,%.3f)\\\\p1}%s" % (line.left, line.top, Convert.text_to_shape(line))
                io.write_line(line)
        """
        # Obtaining information and editing values of style if requested
        obj = ass_text.deep_copy()

        # Editing temporary the style to properly get the shape
        if fscx is not None:
            obj.style.scale_x = fscx
        if fscy is not None:
            obj.style.scale_y = fscy

        # Obtaining font information from style and obtaining shape
        font = Font(obj.style)
        shape = font.text_to_shape(obj.text)
        # Clearing resources to not let overflow errors take over
        del font

        return shape

    @classmethod
    def text_to_clip(cls, ass_text: AssText, an: Alignment, fscx: Optional[float] = None, fscy: Optional[float] = None) -> Shape:
        """Converts text with given style information to an ASS shape, applying some translation/scaling to it since
        it is not possible to position a shape with \\pos() once it is in a clip.

        This is an high level function since it does some additional operations, check text_to_shape for further infromations.

        **Tips:** *You can easily create text masks even for growing/shrinking text without too much effort.*

        Parameters:
            obj (Line, Word, Syllable or Char): An object of class Line, Word, Syllable or Char.
            an (integer, optional): The alignment wanted for the shape.
            fscx (float, optional): The scale_x value for the shape.
            fscy (float, optional): The scale_y value for the shape.

        Returns:
            A Shape object, representing the text with the style format values of the object.

        Examples:
            ..  code-block:: python3

                line = Line.copy(lines[1])
                line.text = "{\\\\an5\\\\pos(%.3f,%.3f)\\\\clip(%s)}%s" % (line.center, line.middle, Convert.text_to_clip(line), line.text)
                io.write_line(line)
        """
        obj = ass_text.deep_copy()

        # Setting default values
        if fscx is None:
            fscx = obj.style.scale_x
        if fscy is None:
            fscy = obj.style.scale_y

        # Obtaining text converted to shape
        shape = cls.text_to_shape(obj, fscx, fscy)

        # Setting mult_x based on alignment
        if an in {1, 4, 7}:
            mult_x = 0
        elif an in {2, 5, 8}:
            mult_x = 1 / 2
        else:
            mult_x = 1

        # Setting mult_y based on alignment
        if an in {1, 2, 3}:
            mult_y = 1
        elif an in {4, 5, 6}:
            mult_y = 1 / 2
        else:
            mult_y = 0

        # Calculating offsets
        cx = obj.left - obj.width * mult_x * (fscx - obj.style.scale_x) / obj.style.scale_x
        cy = obj.top - obj.height * mult_y * (fscy - obj.style.scale_y) / obj.style.scale_y

        shape.move(cx, cy)

        return shape

    @classmethod
    def text_to_pixels(cls, ass_text: AssText, supersampling: int = 8) -> List[Pixel]:
        """| Converts text with given style information to a list of pixel data.
        | A pixel data is a NamedTuple with the attributes 'x' (horizontal position), 'y' (vertical position) and 'alpha' (alpha/transparency/opacity).

        It is highly suggested to create a dedicated style for pixels,
        because you will write less tags for line in your pixels, which means less size for your .ass file.

        | The style suggested is:
        | - **an=7 (very important!);**
        | - bord=0;
        | - shad=0;
        | - For Font informations leave whatever the default is;

        **Tips:** *It allows easy creation of text decaying or light effects.*

        Parameters:
            obj (Line, Word, Syllable or Char): An object of class Line, Word, Syllable or Char.
            supersampling (int): Value used for supersampling. Higher value means smoother and more precise anti-aliasing (and more computational time for generation).

        Returns:
            A list of dictionaries representing each individual pixel of the input text styled.

        Examples:
            ..  code-block:: python3

                line = lines[2].copy()
                line.style = "p"
                p_sh = Shape.rectangle()
                for pixel in Convert.text_to_pixels(line):
                    x, y = math.floor(line.left) + pixel['x'], math.floor(line.top) + pixel['y']
                    alpha = "\\alpha" + Convert.color_alpha_to_ass(pixel['alpha']) if pixel['alpha'] != 255 else ""

                    line.text = "{\\p1\\pos(%d,%d)%s}%s" % (x, y, alpha, p_sh)
                    io.write_line(line)
        """
        shape = Convert.text_to_shape(ass_text)
        shape.move(ass_text.left % 1, ass_text.top % 1)
        return Convert.shape_to_pixels(shape, supersampling)

    @staticmethod
    def shape_to_pixels(shape: Shape, supersampling: int = 8) -> List[Pixel]:
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
        shape.map(lambda x, y: (x * upscale, y * upscale))

        # Bring shape near origin in positive room
        x1, y1, x2, y2 = shape.bounding()
        shift_x, shift_y = -1 * (x1 - x1 % upscale), -1 * (y1 - y1 % upscale)
        shape.move(shift_x, shift_y)

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

        shape.flatten().map(collect_lines)

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
        _, y1, _, y2 = shape.bounding()
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

    @staticmethod
    def image_to_ass(image: Any) -> NoReturn:
        raise NotImplementedError

    @staticmethod
    def image_to_pixels(image: Any) -> NoReturn:
        raise NotImplementedError
