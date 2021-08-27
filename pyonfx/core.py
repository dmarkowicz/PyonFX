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

__all__ = [
    'Meta', 'Style',
    'Line', 'Word', 'Syllable', 'Char'
]

import copy
from abc import ABC
from fractions import Fraction
from typing import Any, Dict, List, Optional, TypeVar, cast

from .colourspace import ASSColor, Opacity
from .convert import ConvertTime
from .font_utility import Font
from .shape import Pixel, Shape
from .types import Alignment

AssTextT = TypeVar('AssTextT', bound='AssText')


class DataCore(ABC):
    """Abstract DataCore object"""

    def __str__(self) -> str:
        return self._pretty_print(self)

    def __repr__(self) -> str:
        return repr(self.__dict__)

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__

    @classmethod
    def _pretty_print(cls, obj: DataCore, indent: int = 0, name: Optional[str] = None) -> str:
        if not name:
            out = " " * indent + f'{obj.__class__.__name__}:\n'
        else:
            out = " " * indent + f'{name}: ({obj.__class__.__name__}):\n'

        # Let's print all this object fields
        indent += 4
        for k, v in obj.__dict__.items():
            if isinstance(v, DataCore):
                # Work recursively to print another object
                out += cls._pretty_print(v, indent, k)
            elif isinstance(v, list):
                v = cast(List[DataCore], v)
                for el in v:
                    # Work recursively to print other objects inside a list
                    out += cls._pretty_print(el, indent, k)
            else:
                # Just print a field of this object
                out += " " * indent + f"{k}: {str(v)}\n"

        return out


class Meta(DataCore):
    """Meta object contains informations about the Ass.

    More info about each of them can be found on http://docs.aegisub.org/manual/Styles
    """
    wrap_style: int
    """Determines how line breaking is applied to the subtitle line"""
    scaled_border_and_shadow: bool
    """Determines if it has to be used script resolution (*True*) or video resolution (*False*) to scale border and shadow"""
    play_res_x: int
    """Video width"""
    play_res_y: int
    """Video height"""
    audio: str
    """Loaded audio path (absolute)"""
    video: str
    """Loaded video path (absolute)"""
    fps: Fraction
    """FrameRate per Second"""


class Style(DataCore):
    """Style object contains a set of typographic formatting rules that is applied to dialogue lines.

    More info about styles can be found on http://docs.aegisub.org/3.2/ASS_Tags/.
    """
    name: str
    """Style name"""
    fontname: str
    """Font name"""
    fontsize: float
    """Font size in points"""
    color1: ASSColor
    alpha1: Opacity
    """Primary color (fill) and transparency"""
    color2: ASSColor
    alpha2: Opacity
    """Secondary color (secondary fill, for karaoke effect) and transparency"""
    color3: ASSColor
    alpha3: Opacity
    """Outline (border) color and transparency"""
    color4: ASSColor
    alpha4: Opacity
    """Shadow color and transparency"""
    bold: bool
    """Font with bold"""
    italic: bool
    """Font with italic"""
    underline: bool
    """Font with underline"""
    strikeout: bool
    """Font with strikeout"""
    scale_x: float
    """Text stretching in the horizontal direction"""
    scale_y: float
    """Text stretching in the vertical direction"""
    spacing: float
    """Horizontal spacing between letters"""
    angle: float
    """Rotation of the text"""
    border_style: bool
    """*True* for opaque box, *False* for standard outline"""
    outline: float
    """Border thickness value"""
    shadow: float
    """How far downwards and to the right a shadow is drawn"""
    _alignment: int
    """Alignment of the text"""
    margin_l: int
    """Distance from the left of the video frame"""
    margin_r: int
    """Distance from the right of the video frame"""
    margin_v: int
    """Distance from the bottom (or top if alignment >= 7) of the video frame"""
    encoding: int
    """Codepage used to map codepoints to glyphs"""

    @property
    def alignment(self) -> int:
        return self._alignment

    @alignment.setter
    def alignment(self, an: Alignment) -> None:
        self._alignment = an

    def an_is_left(self) -> bool:
        return self.alignment in {1, 4, 7}

    def an_is_center(self) -> bool:
        return self.alignment in {2, 5, 8}

    def an_is_right(self) -> bool:
        return self.alignment in {3, 6, 9}

    def an_is_top(self) -> bool:
        return self.alignment in {7, 8, 9}

    def an_is_middle(self) -> bool:
        return self.alignment in {4, 5, 6}

    def an_is_bottom(self) -> bool:
        return self.alignment in {1, 2, 3}


class AssText(DataCore, ABC):
    """Abstract AssText object"""
    i: int
    """Index number"""
    start_time: float
    """Start time (in seconds)"""
    end_time: float
    """End time (in seconds)"""
    duration: float
    """Duration (in seconds)"""
    text: str
    """Text"""
    style: Style
    """Reference to the Style object"""
    meta: Meta
    """Reference toe the Meta object"""
    width: float
    """Text width"""
    height: float
    """Text height"""
    ascent: float
    """Font ascent"""
    descent: float
    """Font descent"""
    internal_leading: float
    """Internal leading"""
    external_leading: float
    """External leading"""
    x: float
    """Text position horizontal (depends on alignment)"""
    y: float
    """Text position vertical (depends on alignment)."""
    left: float
    """Text position left"""
    center: float
    """Text position center"""
    right: float
    """Text position right"""
    top: float
    """Text position top"""
    middle: float
    """Text position middle"""
    bottom: float
    """Text position bottom"""

    def deep_copy(self: AssTextT) -> AssTextT:
        """
        Returns:
            A deep copy of this object
        """
        return copy.deepcopy(self)

    def to_shape(self, fscx: Optional[float] = None, fscy: Optional[float] = None) -> Shape:
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
        obj = self.deep_copy()

        # Editing temporary the style to properly get the shape
        if fscx is not None:
            obj.style.scale_x = fscx
        if fscy is not None:
            obj.style.scale_y = fscy

        # Obtaining font information from style and obtaining shape
        font = Font(obj.style)
        shape = font.text_to_shape(obj.text)
        # Clearing resources to not let overflow errors take over
        del font, obj

        return shape

    def to_clip(self, an: Alignment, fscx: Optional[float] = None, fscy: Optional[float] = None) -> Shape:
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
        obj = self.deep_copy()

        # Setting default values
        if fscx is None:
            fscx = obj.style.scale_x
        if fscy is None:
            fscy = obj.style.scale_y

        # Obtaining text converted to shape
        shape = obj.to_shape(fscx, fscy)

        # Setting mult_x based on alignment
        if an in {1, 4, 7}:
            mult_x = 0.0
        elif an in {2, 5, 8}:
            mult_x = 1 / 2
        else:
            mult_x = 1.0

        # Setting mult_y based on alignment
        if an in {1, 2, 3}:
            mult_y = 1.0
        elif an in {4, 5, 6}:
            mult_y = 1 / 2
        else:
            mult_y = 0.0

        # Calculating offsets
        cx = obj.left - obj.width * mult_x * (fscx - obj.style.scale_x) / obj.style.scale_x
        cy = obj.top - obj.height * mult_y * (fscy - obj.style.scale_y) / obj.style.scale_y

        shape.move(cx, cy)

        del obj

        return shape

    def to_pixels(self, ass_text: AssText, supersampling: int = 8) -> List[Pixel]:
        """| Converts text with given style information to a list of pixel data.
        | A pixel data is a NamedTuple with the attributes 'x' (horizontal position), 'y' (vertical position)
        and 'alpha' (alpha/transparency/opacity).

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
            supersampling (int): Value used for supersampling. Higher value means smoother and more precise anti-aliasing
            (and more computational time for generation).

        Returns:
            A list of NamedTuple representing each individual pixel of the input text styled.

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
        shape = self.to_shape()
        shape.move(ass_text.left % 1, ass_text.top % 1)
        return shape.to_pixels(supersampling)



class Line(AssText):
    """
    Line object contains informations about a single line in the Ass.

    Note:
        (*) = This field is available only if :class:`extended<Ass>` = True
    """
    comment: bool
    """If *True*, this line will not be displayed on the screen"""
    layer: int
    """Layer for the line. Higher layer numbers are drawn on top of lower ones"""
    leadin: float
    """Time between this line and the previous one (in seconds; first line = 1.001) (*)"""
    leadout: float
    """Time between this line and the next one (in seconds; first line = 1.001) (*)"""
    actor: str
    """Actor field"""
    margin_l: int
    """Left margin for this line"""
    margin_r: int
    """Right margin for this line"""
    margin_v: int
    """Vertical margin for this line"""
    effect: str
    """Effect field"""
    raw_text: str
    """Line raw text"""
    words: List[Word]
    """List containing objects :class:`Word` in this line (*)"""
    syls: List[Syllable]
    """List containing objects :class:`Syllable` in this line (if available) (*)"""
    chars: List[Char]
    """List containing objects :class:`Char` in this line (*)"""

    def strip_empty(self, words: bool = True, syls: bool = True, chars: bool = True) -> Line:
        if words:
            self.words = self.strip_obj_empty(self.words)
        if syls:
            self.syls = self.strip_obj_empty(self.syls)
        if chars:
            self.chars = self.strip_obj_empty(self.chars)
        return self

    @staticmethod
    def strip_obj_empty(obj: List[AssTextT]) -> List[AssTextT]:
        return [o for o in obj if o.text.strip() and o.duration > 0]

    def compose_ass_line(self) -> str:
        ass_line = "Comment: " if self.comment else "Dialogue: "
        elements: List[Any] = [
            self.layer,
            ConvertTime.seconds2assts(self.start_time, self.meta.fps),
            ConvertTime.seconds2assts(self.end_time, self.meta.fps),
            self.style.name, self.actor,
            self.margin_l, self.margin_r, self.margin_v,
            self.effect, self.text
        ]
        return ass_line + ','.join(map(str, elements)) + '\n'


class Word(AssText):
    """
    Word object contains informations about a single word of a line in the Ass.

    A word can be defined as some text with some optional space before or after.
    (e.g.: In the string "What a beautiful world!", "beautiful" and "world" are both distinct words).
    """
    prespace: int
    """Word free space before text"""
    postspace: int
    """Word free space after text"""


class WordElement(Word, ABC):
    """Abstract WordElement class"""
    word_i: int
    """Word index (e.g.: In line text ``Hello PyonFX users!``, letter "u" will have word_i=2)"""
    inline_fx: str
    """Inline effect (marked as \\-EFFECT in karaoke-time)"""


class Syllable(WordElement):
    """
    Syllable object contains informations about a single syl of a line in the Ass.

    A syl can be defined as some text after a karaoke tag (k, ko, kf)
    (e.g.: In ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, "Pyon" and "FX" are distinct syllables),
    """
    tags: str
    """All the remaining tags before syl text apart \\k ones"""


class Char(WordElement):
    """
    Char object contains informations about a single char of a line in the Ass.

    A char is defined by some text between two karaoke tags (k, ko, kf).
    """
    syl_i: int
    """Char syl index (e.g.: In line text ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, letter "F" will have syl_i=3)"""
    syl_char_i: int
    """Char invidual syl index (e.g.: In line text ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, letter "e"
    of "users will have syl_char_i=2)"""
