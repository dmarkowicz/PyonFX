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
"""Main core module"""
from __future__ import annotations

__all__ = [
    'Meta', 'Style',
    'Line', 'Word', 'Syllable', 'Char',
    'Ass'
]

import copy
import os
import re
import subprocess
import sys
import time
import warnings
from abc import ABC
from collections import UserList
from fractions import Fraction
from pathlib import Path
from pprint import pformat
from typing import (
    Any, Dict, Hashable, Iterable, Iterator, List, Literal, Mapping, NamedTuple, Optional, TextIO,
    Tuple, TypeVar, cast, overload
)

from more_itertools import zip_offset

from .colourspace import ASSColor, Opacity
from .convert import ConvertTime
from .exception import MatchNotFoundError
from .font import Font
from .shape import Pixel, Shape
from .types import Alignment, AnyPath, AutoSlots, NamedMutableSequence

_AssTextT = TypeVar('_AssTextT', bound='_AssText')


class Ass:
    """Initialisation class containing all the information about an ASS file"""
    fps: Fraction

    meta: Meta
    styles: List[Style]
    lines: List[Line]

    _path_output: Optional[Path]
    _output: List[str]
    _output_lines: List[str]
    _output_extradata: List[str]

    _ptime: float

    def __init__(self, path_input: AnyPath | TextIO, path_output: Optional[AnyPath] = None, /,
                 fps: Fraction | float = Fraction(24000, 1001),
                 comment_original: bool = True, extended: bool = True, vertical_kanji: bool = False) -> None:
        """
        :param _path_input:          Input file path
        :param _path_output:         Output file path, defaults to None
        :param fps:                 Framerate Per Second of the ASS file, defaults to Fraction(24000, 1001)
        :param comment_original:    Keep original lines at the beginning of the output ass and comment them, defaults to True
        :param extended:            Calculate more informations from lines, defaults to True
        :param vertical_kanji:      Line text with alignment 4, 5 or 6 will be positioned vertically
                                    Additionally, ``line`` fields will be re-calculated based on the re-positioned ``line.chars``,
                                    defaults to False
        """
        self._ptime = time.time()

        self._path_output = Path(path_output).resolve() if path_output else None
        self.fps = fps if isinstance(fps, Fraction) else Fraction(fps)
        self._output = []
        self._output_lines = []
        self._output_extradata = []

        if isinstance(path_input, TextIO):
            lines_file = path_input.read()
        else:
            with open(path_input, "r", encoding="utf-8-sig") as file:
                lines_file = file.read()

        # Find section pattern
        matches = re.finditer(r'(^\[[^\]]*])', lines_file, re.MULTILINE)
        sections = [_Section(m.group(0), *m.span(0)) for m in matches]

        for sec1, sec2 in zip_offset(sections, sections, offsets=(0, 1), longest=True, fillvalue=_Section('', start=None)):
            sec1.text = lines_file[sec1.end:sec2.start]

        self.meta = Meta.from_text(
            ''.join(
                s.text for s in sections
                if s.name in {'[Script Info]', '[Aegisub Project Garbage]'}),
            self.fps
        )
        self.styles = [
            Style.from_text(stext)
            for s in sections
            if s.name == '[V4+ Styles]'
            for stext in s.text.strip().splitlines()[1:]
        ]
        self.lines = [
            Line.from_text(ltext, i, self.fps, self.meta, self.styles)
            for s in sections
            if s.name == '[Events]'
            for i, ltext in enumerate(s.text.strip().splitlines()[1:])
        ]

        if not extended:
            return None

        lines_by_styles: Dict[str, List[Line]] = {style.name: [] for style in self.styles}
        for line in self.lines:
            # Add dialog text sizes and positions (if possible)
            try:
                line_style = line.style
            except AttributeError:
                warnings.warn(f'Line {line.i} is using an undefined style, skipping...', Warning)
                continue
            lines_by_styles[line_style.name].append(line)

            font = Font(line_style)
            line.add_data(font)
            line.add_words(font)
            line.add_syls(font, vertical_kanji)
            line.add_chars(font, vertical_kanji)

        # Add durations between dialogs
        fps = float(self.fps)
        default_lead = 1 / fps * round(fps)
        for liness in lines_by_styles.values():
            for preline, curline, postline in zip_offset(liness, liness, liness, offsets=(-1, 0, 1), longest=True):
                if not curline:
                    continue
                curline.leadin = default_lead if not preline else curline.start_time - preline.end_time
                curline.leadout = default_lead if not postline else postline.start_time - curline.end_time

    def get_data(self) -> Tuple[Meta, List[Style], List[Line]]:
        """
        Utility function to easily retrieve meta, styles and lines.

        :return:            :attr:`meta`, :attr:`styles` and :attr:`lines`
        """
        return self.meta, self.styles, self.lines

    def write_line(self, line: Line) -> None:
        """
        Appends a line to the output list that later on will be written to the output file
        when calling save().
        Use it whenever you've prepared a line, it will not impact performance
        since you will not actually write anything until :func:`save` will be called.

        :param line:        Line object
        """
        self._output_lines.append(line.compose_ass_line())

    def save(self, lines: Optional[Iterable[Line]] = None, quiet: bool = False) -> None:
        """
        Write everything inside the output list to a file.

        :param lines:       Additional Line objects to be written
        :param quiet:       Don't show message, defaults to False
        """
        if not self._path_output:
            raise ValueError('path_output hasn\'t been specified in the constructor')

        with self._path_output.open("w", encoding="utf-8-sig") as file:
            file.writelines(
                self._output
                + self._output_lines
                + ([line.compose_ass_line() for line in lines] if lines else [])
                + ["\n"]
            )
            if self._output_extradata:
                file.write("\n[Aegisub Extradata]\n")
                file.writelines(self._output_extradata)

        if not quiet:
            print(
                f"Produced lines: {len(self._output_lines)}\n"
                f"Process duration (in seconds): {round(time.time() - self._ptime, ndigits=3)}"
            )

    def open_aegisub(self) -> None:
        """
        Open the output (specified in _path_output during the initialisation class) with Aegisub.
        """
        if not self._path_output:
            raise ValueError('path_output hasn\'t been specified in the constructor')
        # Check if it was saved
        if not self._path_output.exists():
            warnings.warn(
                f'{self.__class__.__name__}: "_path_output" not found!', Warning
            )
        else:
            if sys.platform == "win32":
                os.startfile(self._path_output)
            else:
                try:
                    subprocess.call(["aegisub", self._path_output])
                except FileNotFoundError:
                    warnings.warn("Aegisub not found!", Warning)

    def open_mpv(self, video_path: AnyPath | None = None,
                 video_start: Optional[str] = None, full_screen: bool = False) -> None:
        """
        Open the output (specified in _path_output during the initialisation class)
        in softsub with MPV player.
        You should add MPV in your PATH (check https://pyonfx.readthedocs.io/en/latest/quick%20start.html#installation-extra-step).

        :param video_path:          Video path. If not specified it will use the path in meta.video, defaults to None
        :param video_start:         Start time for the video (more info: https://mpv.io/manual/master/#options-start)
                                    If not specified, 0 is automatically taken, defaults to None
        :param full_screen:         Launch MPV in full screen, defaults to False
        """
        if not self._path_output:
            raise ValueError('path_output hasn\'t been specified in the constructor')
        # Check if it was saved
        if not self._path_output.exists():
            warnings.warn(
                f'{self.__class__.__name__}: "_path_output" not found!', Warning
            )
            return None

        # Check if mpv is usable
        if self.meta.video.startswith("?dummy") and not video_path:
            warnings.warn(
                'Cannot use MPV (if you have it in your PATH) for file preview, since your .ass contains a dummy video.\n'
                'You can specify a new video source using video_path parameter, check the documentation of the function.',
                Warning
            )
        else:
            # Setting up the command to execute
            cmd = ["mpv"]

            if video_path:
                cmd.append(str(video_path))
            else:
                cmd.append(self.meta.video)
            if video_start:
                cmd.append("--start=" + video_start)
            if full_screen:
                cmd.append("--fs")

            cmd.append("--sub-file=" + str(self._path_output))

            try:
                subprocess.call(cmd)
            except FileNotFoundError:
                warnings.warn(
                    "MPV not found in your environment variables.\n"
                    "Please refer to the documentation's \"Quick Start\" section if you don't know how to solve it.",
                    Warning
                )


class PList(UserList[_AssTextT]):
    """PyonFX list"""

    def __init__(self, __iterable: Iterable[_AssTextT] | None = None, /) -> None:
        """
        If no argument is given, the constructor creates a new empty list.

        :param iterable:            Iterable object, defaults to None
        """
        super().__init__(__iterable)

    @overload
    def strip_empty(self, return_new: Literal[False] = False) -> None:
        """
        Removes objects with empty text or a duration of 0

        :param return_new:          If False, works on the current object, defaults to False
        """
        ...

    @overload
    def strip_empty(self, return_new: Literal[True]) -> PList[_AssTextT]:
        """
        Removes objects with empty text or a duration of 0

        :param return_new:          If True, returns a new PList
        """
        ...

    def strip_empty(self, return_new: bool = False) -> None | PList[_AssTextT]:
        for x in (data := self.copy() if return_new else self.data):
            if not (x.text.strip() != '' and x.duration > 0):
                data.remove(x)
        return self.__class__(data) if return_new else None


class DataCore(AutoSlots, Hashable, Mapping[str, Any], ABC, empty_slots=True):
    """Abstract DataCore object"""

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __getitem__(self, __k: str) -> Any:
        try:
            return self.__getattribute__(__k)
        except AttributeError:
            return None

    def __iter__(self) -> Iterator[str]:
        for k in self.__slots__:
            yield k

    def __len__(self) -> int:
        return self.__slots__.__len__()

    def __str__(self) -> str:
        return self._pretty_print(self)

    def __repr__(self) -> str:
        return pformat(dict(self))

    def _asdict(self) -> Dict[str, Any]:
        return {k: (dict(v) if isinstance(v, DataCore) else v) for k, v in self.items()}

    def _pretty_print(self, obj: DataCore, indent: int = 0, name: Optional[str] = None) -> str:
        if not name:
            out = " " * indent + f'{obj.__class__.__name__}:\n'
        else:
            out = " " * indent + f'{name}: ({obj.__class__.__name__}):\n'

        indent += 4
        for k, v in obj.items():
            if isinstance(v, DataCore):
                # Work recursively to print another object
                out += self._pretty_print(v, indent, k)
            elif isinstance(v, PList):
                for el in v:
                    # Work recursively to print other objects inside a list
                    out += self._pretty_print(el, indent, k)
            else:
                # Just print a field of this object
                out += " " * indent + f"{k}: {str(v)}\n"
        return out


class Meta(DataCore):
    """
    Meta object contains informations about the Ass.

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

    @classmethod
    def from_text(cls, text: str | List[str], fps: Fraction) -> Meta:
        self = cls()
        self.fps = fps
        for line in (text.splitlines(False) if isinstance(text, str) else text):
            if valm := re.match(r"WrapStyle: *?(\d+)$", line):
                self.wrap_style = int(valm[1].strip())
            elif valm := re.match(r"ScaledBorderAndShadow: *?(.+)$", line):
                self.scaled_border_and_shadow = valm[1].strip() == "yes"
            elif valm := re.match(r"PlayResX: *?(\d+)$", line):
                self.play_res_x = int(valm[1].strip())
            elif valm := re.match(r"PlayResY: *?(\d+)$", line):
                self.play_res_y = int(valm[1].strip())
            elif valm := re.match(r"Audio File: *?(.*)$", line):
                self.audio = str(valm)
            elif valm := re.match(r"Video File: *?(.*)$", line):
                self.video = str(valm)
        return self


class Style(DataCore):
    """
    Style object contains a set of typographic formatting rules that is applied to dialogue lines.

    More info about styles can be found on http://docs.aegisub.org/3.2/ASS_Tags/.
    """
    name: str
    """Style name"""
    fontname: str
    """Font name"""
    fontsize: float
    """Font size in points"""
    color1: ASSColor
    """Primary color (fill)"""
    alpha1: Opacity
    """Transparency of color1"""
    color2: ASSColor
    """Secondary color (secondary fill, for karaoke effect)"""
    alpha2: Opacity
    """Transparency of color2"""
    color3: ASSColor
    """Outline (border) color"""
    alpha3: Opacity
    """Transparency color3"""
    color4: ASSColor
    """Shadow color"""
    alpha4: Opacity
    """Transparency of color4"""
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
    alignment: int
    """Alignment of the text. Must be in the range 1 <= an <= 9"""
    margin_l: int
    """Distance from the left of the video frame"""
    margin_r: int
    """Distance from the right of the video frame"""
    margin_v: int
    """Distance from the bottom (or top if alignment >= 7) of the video frame"""
    encoding: int
    """Codepage used to map codepoints to glyphs"""

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

    @classmethod
    def from_text(cls, text: str) -> Style:
        self = cls()

        if not (style_match := re.match(r"Style: (.+?)$", text)):
            raise MatchNotFoundError
        # Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,
        # Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,
        # BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        style = style_match[1].split(",")

        self.name = str(style[0])

        self.fontname = str(style[1])
        self.fontsize = float(style[2])

        self.color1 = ASSColor(f"&H{style[3][4:]}&")
        self.color2 = ASSColor(f"&H{style[4][4:]}&")
        self.color3 = ASSColor(f"&H{style[5][4:]}&")
        self.color4 = ASSColor(f"&H{style[6][4:]}&")

        self.alpha1 = Opacity.from_ass_val(f"{style[3][:4]}&")
        self.alpha2 = Opacity.from_ass_val(f"{style[4][:4]}&")
        self.alpha3 = Opacity.from_ass_val(f"{style[5][:4]}&")
        self.alpha4 = Opacity.from_ass_val(f"{style[6][:4]}&")

        self.bold = style[7] == "-1"
        self.italic = style[8] == "-1"
        self.underline = style[9] == "-1"
        self.strikeout = style[10] == "-1"

        self.scale_x = float(style[11])
        self.scale_y = float(style[12])

        self.spacing = float(style[13])
        self.angle = float(style[14])

        self.border_style = style[15] == "3"
        self.outline = float(style[16])
        self.shadow = float(style[17])

        self.alignment = int(style[18])
        self.margin_l = int(style[19])
        self.margin_r = int(style[20])
        self.margin_v = int(style[21])

        self.encoding = int(style[22])

        return self


class _PositionedText(DataCore, ABC, empty_slots=True):
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

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _PositionedText.__annotations__:
            value = round(value, 3)
        return super().__setattr__(name, value)


class _AssText(_PositionedText, ABC, empty_slots=True):
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
    """Reference to the Meta object"""
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

    def __copy__(self: _AssTextT) -> _AssTextT:
        obj = self.__class__()
        for k, v in self.items():
            setattr(obj, k, v)
        return obj

    def __deepcopy__(self: _AssTextT, *args: Any) -> _AssTextT:
        obj = self.__class__()
        for k, v in self.items():
            setattr(obj, k, copy.deepcopy(v))
        return obj

    def deep_copy(self: _AssTextT) -> _AssTextT:
        """
        :return:            A deep copy of this object
        """
        return copy.deepcopy(self)

    def shallow_copy(self: _AssTextT) -> _AssTextT:
        """
        :return:            A shallow copy of this object
        """
        return copy.copy(self)

    def copy(self: _AssTextT) -> _AssTextT:
        """
        :return:            A shallow copy of this object
        """
        return self.shallow_copy()

    def to_shape(self, fscx: Optional[float] = None, fscy: Optional[float] = None) -> Shape:
        """
        Converts text with given style information to an ASS shape.
        **Tips:** *You can easily create impressive deforming effects.*

        Examples:
            ..  code-block:: python

                l = line.deep_copy()
                l.text = f"{\\\\an7\\\\pos({line.left},{line.top})\\\\p1}{line.to_shape()}"
                io.write_line(l)

        :param fscx:        The scale_x value for the shape, defaults to None
        :param fscy:        The scale_y value for the shape, defaults to None
        :return:            A Shape object, representing the text with the style format values
                            of the object
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
        """
        Converts text with given style information to an ASS shape, applying some translation/scaling to it
        since it is not possible to position a shape with \\pos() once it is in a clip.
        **Tips:** *You can easily create text masks even for growing/shrinking text without too much effort.*

        Examples:
            ..  code-block:: python

                l = line.deep_copy()
                l.text = f"{\\\\an5\\\\pos({line.center},{line.middle})\\\\clip({line.to_clip()})}{line.text}"
                io.write_line(l)

        :param an:          Alignment wanted for the shape
        :param fscx:        The scale_x value for the shape, defaults to None
        :param fscy:        The scale_y value for the shape, defaults to None
        :return:            A Shape object, representing the text with the style format values of the object
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

    def to_pixels(self, supersampling: int = 4, anti_aliasing: bool = True) -> List[Pixel]:
        """
        Converts text with given style information to a list of Pixel.
        It is strongly recommended to create a dedicated style for pixels,
        thus, you will write less tags for line in your pixels,
        which means less size for your .ass file.

        Style suggested as an=7, bord=0, shad=0

        :param supersampling:   Supersampling value.
                                Higher value means smoother and more precise anti-aliasing, defaults to 4
        :return:                A list of Pixel representing each individual pixel of the input text styled.
        """
        return self.to_shape().to_pixels(supersampling, anti_aliasing)


class Line(_AssText):
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
    words: PList[Word]
    """List containing objects :class:`Word` in this line (*)"""
    syls: PList[Syllable]
    """List containing objects :class:`Syllable` in this line (if available) (*)"""
    chars: PList[Char]
    """List containing objects :class:`Char` in this line (*)"""

    @classmethod
    def from_text(cls, text: str | List[str], i: int, fps: Fraction,
                  meta: Optional[Meta] = None, styles: Optional[Iterable[Style]] = None) -> Line:
        self = cls()

        for line in (text.splitlines(False) if isinstance(text, str) else text):
            # Analysing line
            if not (anal_line := re.match(r"(Dialogue|Comment): (.+?)$", line)):
                raise MatchNotFoundError
            # Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            self.i = i

            self.comment = anal_line[1] == "Comment"
            linesplit = anal_line[2].split(",")

            self.layer = int(linesplit[0])

            self.start_time = ConvertTime.assts2seconds(linesplit[1], fps)
            self.start_time = ConvertTime.bound_to_frame(self.start_time, fps)
            self.end_time = ConvertTime.assts2seconds(linesplit[2], fps)
            self.end_time = ConvertTime.bound_to_frame(self.end_time, fps)
            self.duration = self.end_time - self.start_time

            for style in (styles if styles else []):
                if style.name == linesplit[3]:
                    self.style = style
                    break
            if meta:
                self.meta = meta
            self.actor = linesplit[4]

            self.margin_l = int(linesplit[5])
            self.margin_r = int(linesplit[6])
            self.margin_v = int(linesplit[7])

            self.effect = linesplit[8]

            self.raw_text = ",".join(linesplit[9:])
            self.text = self.raw_text

        return self

    def add_data(self, font: Font) -> None:
        self.text = re.sub(r"\{.*?\}", "", self.raw_text)

        self.width, self.height = font.text_extents(self.text)
        self.ascent, self.descent, self.internal_leading, self.external_leading = font.metrics

        try:
            play_res_x = self.meta.play_res_x
            play_res_y = self.meta.play_res_y
            style = self.style
        except AttributeError:
            return None

        # Horizontal position
        margin_l = self.margin_l if self.margin_l != 0 else style.margin_l
        margin_r = self.margin_r if self.margin_r != 0 else style.margin_r
        if style.an_is_left():
            self.left = margin_l
            self.center = self.left + self.width / 2
            self.right = self.left + self.width
            self.x = self.left
        elif style.an_is_center():
            self.left = play_res_x / 2 - self.width / 2 + margin_l / 2 - margin_r / 2
            self.center = self.left + self.width / 2
            self.right = self.left + self.width
            self.x = self.center
        else:
            self.left = play_res_x - margin_r - self.width
            self.center = self.left + self.width / 2
            self.right = self.left + self.width
            self.x = self.right

        # Vertical position
        if style.an_is_top():
            self.top = self.margin_v if self.margin_v != 0 else style.margin_v
            self.middle = self.top + self.height / 2
            self.bottom = self.top + self.height
            self.y = self.top
        elif style.an_is_middle():
            self.top = play_res_y / 2 - self.height / 2
            self.middle = self.top + self.height / 2
            self.bottom = self.top + self.height
            self.y = self.middle
        else:
            self.top = play_res_y - (self.margin_v if self.margin_v != 0 else style.margin_v) - self.height
            self.middle = self.top + self.height / 2
            self.bottom = self.top + self.height
            self.y = self.bottom

    def add_words(self, font: Font) -> None:
        # Adding words
        self.words = PList()

        presp_txt_postsp = re.findall(r"(\s*)([^\s]+)(\s*)", self.text)
        presp_txt_postsp = cast(List[Tuple[str, str, str]], presp_txt_postsp)

        for wi, (prespace, word_text, postspace) in enumerate(presp_txt_postsp):
            word = Word()

            word.i = wi

            word.start_time = self.start_time
            word.end_time = self.end_time
            word.duration = self.duration

            word.meta = self.meta
            word.style = self.style
            word.text = word_text

            word.prespace = len(prespace)
            word.postspace = len(postspace)

            word.width, word.height = font.text_extents(word.text)
            word.ascent, word.descent, word.internal_leading, word.external_leading = font.metrics

            self.words.append(word)

        # Calculate word positions with all words data already available
        try:
            play_res_x = self.meta.play_res_x
            play_res_y = self.meta.play_res_y
            style = self.style
        except AttributeError:
            return None

        # Calculating space width and saving spacing
        space_width = font.text_extents(" ")[0]

        if style.an_is_top() or style.an_is_bottom():
            cur_x = self.left
            for word in self.words:
                # Horizontal position
                cur_x += word.prespace * space_width + self.style.spacing
                word.left = cur_x
                word.center = word.left + word.width / 2
                word.right = word.left + word.width

                if self.style.an_is_left():
                    word.x = word.left
                elif self.style.an_is_center():
                    word.x = word.center
                else:
                    word.x = word.right

                # Vertical position
                word.top = self.top
                word.middle = self.middle
                word.bottom = self.bottom
                word.y = self.y

                # Updating cur_x
                cur_x += word.width + word.postspace * (space_width + self.style.spacing) + self.style.spacing
        else:
            max_width, sum_height = 0.0, 0.0
            for word in self.words:
                max_width = max(max_width, word.width)
                sum_height += word.height

            cur_y = x_fix = play_res_y / 2 - sum_height / 2
            for word in self.words:
                # Horizontal position
                x_fix = (max_width - word.width) / 2

                if self.style.alignment == 4:
                    word.left = self.left + x_fix
                    word.center = word.left + word.width / 2
                    word.right = word.left + word.width
                    word.x = word.left
                elif self.style.alignment == 5:
                    word.left = play_res_x / 2 - word.width / 2
                    word.center = word.left + word.width / 2
                    word.right = word.left + word.width
                    word.x = word.center
                else:
                    word.left = self.right - word.width - x_fix
                    word.center = word.left + word.width / 2
                    word.right = word.left + word.width
                    word.x = word.right

                # Vertical position
                word.top = cur_y
                word.middle = word.top + word.height / 2
                word.bottom = word.top + word.height
                word.y = word.middle
                # Updating cur_y
                cur_y += word.height
        return None

    def add_syls(self, font: Font, vertical_kanji: bool = False) -> None:
        # Adding syls
        si = 0
        last_time = 0.0
        inline_fx = ""
        syl_tags_pattern = re.compile(r"(.*?)\\[kK][of]?(\d+)(.*)")

        self.syls = PList()
        for tc in self._text_chunks:
            # If we don't have at least one \k tag, everything is invalid
            if not syl_tags_pattern.match(tc.tags):
                self.syls.clear()
                break

            posttags = tc.tags
            syls_in_text_chunk: List[Syllable] = []
            while 1:
                # Are there \k in posttags?
                tags_syl = syl_tags_pattern.match(posttags)

                if not tags_syl:
                    # Append all the temporary syls, except last one
                    for syl in syls_in_text_chunk[:-1]:
                        curr_inline_fx = re.search(r"\\\-([^\\]+)", syl.tags)
                        if curr_inline_fx:
                            inline_fx = curr_inline_fx[1]
                        syl.inline_fx = inline_fx

                        # Hidden syls are treated like empty syls
                        syl.prespace, syl.text, syl.postspace = 0, "", 0

                        syl.width, syl.height = font.text_extents("")
                        syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font.metrics

                        self.syls.append(syl)

                    # Append last syl
                    syl = syls_in_text_chunk[-1]
                    syl.tags += posttags

                    curr_inline_fx = re.search(r"\\\-([^\\]+)", syl.tags)
                    if curr_inline_fx:
                        inline_fx = curr_inline_fx[1]
                    syl.inline_fx = inline_fx

                    if tc.text.isspace():
                        syl.prespace, syl.text, syl.postspace = 0, tc.text, 0
                    else:
                        if pstxtps := re.match(r"(\s*)(.*?)(\s*)$", tc.text):
                            prespace, syl.text, postspace = pstxtps.groups()
                            syl.prespace, syl.postspace = len(prespace), len(postspace)

                    syl.width, syl.height = font.text_extents(syl.text)
                    syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font.metrics

                    self.syls.append(syl)
                    break

                pretags, kdur, posttags = tags_syl.groups()

                # Create a Syllable object
                syl = Syllable()

                syl.start_time = last_time
                # kdur is in centiseconds
                # Converting in seconds...
                syl.end_time = last_time + int(kdur) / 100
                syl.duration = int(kdur) / 100

                try:
                    syl.style = self.style
                except AttributeError:
                    break
                try:
                    syl.meta = self.meta
                except AttributeError:
                    break
                syl.tags = pretags

                syl.i = si
                if tc.word_i is not None:
                    syl.word_i = tc.word_i

                syls_in_text_chunk.append(syl)

                # Update working variable
                si += 1
                last_time = syl.end_time

        # Calculate syllables positions with all syllables data already available
        try:
            _ = self.meta, self.style
        except AttributeError:
            return None

        space_width = font.text_extents(" ").width

        if self.style.an_is_top() or self.style.an_is_bottom() or not vertical_kanji:
            cur_x = self.left
            for syl in self.syls:
                cur_x += syl.prespace * (space_width + self.style.spacing)

                # Horizontal position
                syl.left = cur_x
                syl.center = syl.left + syl.width / 2
                syl.right = syl.left + syl.width

                if self.style.an_is_left():
                    syl.x = syl.left
                elif self.style.an_is_center():
                    syl.x = syl.center
                else:
                    syl.x = syl.right

                cur_x += syl.width + syl.postspace * (space_width + self.style.spacing) + self.style.spacing

                # Vertical position
                syl.top = self.top
                syl.middle = self.middle
                syl.bottom = self.bottom
                syl.y = self.y

        # Kanji vertical position
        else:
            max_width, sum_height = 0.0, 0.0
            for syl in self.syls:
                max_width = max(max_width, syl.width)
                sum_height += syl.height

            cur_y = self.meta.play_res_y / 2 - sum_height / 2

            for syl in self.syls:
                # Horizontal position
                x_fix = (max_width - syl.width) / 2
                if self.style.alignment == 4:
                    syl.left = self.left + x_fix
                    syl.center = syl.left + syl.width / 2
                    syl.right = syl.left + syl.width
                    syl.x = syl.left
                elif self.style.alignment == 5:
                    syl.left = self.center - syl.width / 2
                    syl.center = syl.left + syl.width / 2
                    syl.right = syl.left + syl.width
                    syl.x = syl.center
                else:
                    syl.left = self.right - syl.width - x_fix
                    syl.center = syl.left + syl.width / 2
                    syl.right = syl.left + syl.width
                    syl.x = syl.right

                # Vertical position
                syl.top = cur_y
                syl.middle = syl.top + syl.height / 2
                syl.bottom = syl.top + syl.height
                syl.y = syl.middle
                cur_y += syl.height
        return None

    def add_chars(self, font: Font, vertical_kanji: bool = False) -> None:
        # Adding chars
        self.chars = PList()

        # If we have syls in line, we prefert to work with them to provide more informations
        if not self.syls:
            if not self.words:
                return None
            words_or_syls = self.syls
        else:
            words_or_syls = self.words

        # Getting chars
        for char_index, el in enumerate(words_or_syls):
            el_text = "{}{}{}".format(" " * el.prespace, el.text, " " * el.postspace)
            for ci, (prespace, char_text, postspace) in enumerate(
                zip_offset(el_text, el_text, el_text, offsets=(-1, 0, 1), longest=True, fillvalue='')
            ):
                if not char_text:
                    continue
                char = Char()
                char.i = char_index
                char_index += 1

                # If we're working with syls, we can add some indexes
                if isinstance(el, Syllable):
                    char.word_i = el.word_i
                    char.syl_i = el.i
                    char.syl_char_i = ci
                else:
                    char.word_i = el.i

                # Adding last fields based on the existance of syls or not
                char.start_time = el.start_time
                char.end_time = el.end_time
                char.duration = el.duration

                char.meta = self.meta
                char.style = self.style
                char.text = char_text

                char.prespace = int(prespace.isspace())
                char.postspace = int(postspace.isspace())

                char.width, char.height = font.text_extents(char.text)
                char.ascent, char.descent, char.internal_leading, char.external_leading = font.metrics

                self.chars.append(char)

        # Calculate character positions with all characters data already available
        try:
            _ = self.meta, self.style
        except AttributeError:
            return None

        if self.style.an_is_top() or self.style.an_is_bottom() or not vertical_kanji:
            cur_x = self.left
            for char in self.chars:
                # Horizontal position
                char.left = cur_x
                char.center = char.left + char.width / 2
                char.right = char.left + char.width

                if self.style.an_is_left():
                    char.x = char.left
                if self.style.an_is_center():
                    char.x = char.center
                else:
                    char.x = char.right

                cur_x += char.width + self.style.spacing

                # Vertical position
                char.top = self.top
                char.middle = self.middle
                char.bottom = self.bottom
                char.y = self.y
        else:
            max_width, sum_height = 0.0, 0.0
            for char in self.chars:
                max_width = max(max_width, char.width)
                sum_height += char.height

            cur_y = x_fix = self.meta.play_res_y / 2 - sum_height / 2

            # Fixing line positions
            self.top = cur_y
            self.middle = self.meta.play_res_y / 2
            self.bottom = self.top + sum_height
            self.width = max_width
            self.height = sum_height
            if self.style.alignment == 4:
                self.center = self.left + max_width / 2
                self.right = self.left + max_width
            elif self.style.alignment == 5:
                self.left = self.center - max_width / 2
                self.right = self.left + max_width
            else:
                self.left = self.right - max_width
                self.center = self.left + max_width / 2

            for char in self.chars:
                # Horizontal position
                x_fix = (max_width - char.width) / 2
                if self.style.alignment == 4:
                    char.left = self.left + x_fix
                    char.center = char.left + char.width / 2
                    char.right = char.left + char.width
                    char.x = char.left
                elif self.style.alignment == 5:
                    char.left = self.meta.play_res_x / 2 - char.width / 2
                    char.center = char.left + char.width / 2
                    char.right = char.left + char.width
                    char.x = char.center
                else:
                    char.left = self.right - char.width - x_fix
                    char.center = char.left + char.width / 2
                    char.right = char.left + char.width
                    char.x = char.right

                # Vertical position
                char.top = cur_y
                char.middle = char.top + char.height / 2
                char.bottom = char.top + char.height
                char.y = char.middle
                cur_y += char.height

    def compose_ass_line(self) -> str:
        """Make an ASS line suitable for writing into ASS file"""
        ass_line = "Comment: " if self.comment else "Dialogue: "
        elements: List[Any] = [
            self.layer,
            ConvertTime.seconds2assts(self.start_time, self.meta.fps),
            ConvertTime.seconds2assts(self.end_time, self.meta.fps),
            self.style.name, self.actor,
            self.margin_l, self.margin_r, self.margin_v,
            self.effect, self.text
        ]
        return ass_line + ','.join(str(el) for el in elements) + '\n'

    @property
    def _text_chunks(self) -> List[_TextChunk]:
        # Search for dialog's text chunks, to later create syllables
        # A text chunk is a text with one or more {tags} preceding it
        # Tags can be some text or empty string

        text_chunks = []
        tag_pattern = re.compile(r"(\{.*?\})+")
        tag = tag_pattern.search(self.raw_text)
        word_i = 0

        if not tag:
            # No tags found
            text_chunks.append(_TextChunk('', self.raw_text))
        else:
            # First chunk without tags?
            if tag.start() != 0:
                text_chunks.append(
                    _TextChunk('', self.raw_text[0:tag.start()])
                )

            # Searching for other tags
            while 1:
                next_tag = tag_pattern.search(self.raw_text, tag.end())
                chk = _TextChunk(
                    self.raw_text[tag.start() + 1: tag.end() - 1].replace("}{", ""),
                    self.raw_text[tag.end(): (next_tag.start() if next_tag else None)],
                    word_i
                )
                text_chunks.append(chk)

                # If there are some spaces after text, then we're at the end of the current word
                if re.match(r"(.*?)(\s+)$", chk.text):
                    word_i += 1

                if not next_tag:
                    break
                tag = next_tag

        return text_chunks


class Word(_AssText):
    """
    Word object contains informations about a single word of a line in the Ass.

    A word can be defined as some text with some optional space before or after.
    (e.g.: In the string "What a beautiful world!", "beautiful" and "world" are both distinct words).
    """
    prespace: int
    """Word free space before text"""
    postspace: int
    """Word free space after text"""


class _WordElement(Word, ABC, empty_slots=True):
    """Abstract WordElement class"""
    word_i: int
    """Word index (e.g.: In line text ``Hello PyonFX users!``, letter "u" will have word_i=2)"""
    inline_fx: str
    """Inline effect (marked as \\-EFFECT in karaoke-time)"""


class Syllable(_WordElement):
    """
    Syllable object contains informations about a single syl of a line in the Ass.

    A syl can be defined as some text after a karaoke tag (k, ko, kf)
    (e.g.: In ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, "Pyon" and "FX" are distinct syllables),
    """
    tags: str
    """All the remaining tags before syl text apart \\k ones"""


class Char(_WordElement):
    """
    Char object contains informations about a single char of a line in the Ass.

    A char is defined by some text between two karaoke tags (k, ko, kf).
    """
    syl_i: int
    """Char syl index (e.g.: In line text ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, letter "F" will have syl_i=3)"""
    syl_char_i: int
    """Char invidual syl index (e.g.: In line text ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, letter "e"
    of "users will have syl_char_i=2)"""


class _TextChunk(NamedTuple):
    tags: str
    text: str
    word_i: Optional[int] = None


class _Section(NamedMutableSequence[Any]):
    name: str
    start: int
    end: Optional[int]
    text: str
