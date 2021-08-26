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

from fractions import Fraction
from typing import List


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
