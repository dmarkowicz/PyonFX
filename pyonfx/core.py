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
from abc import ABC
from collections import UserList, defaultdict
from fractions import Fraction
from functools import lru_cache
from pprint import pformat
from typing import (
    TYPE_CHECKING, Any, DefaultDict, Dict, Iterable, Iterator, List, Literal, Optional, Tuple,
    TypeVar, Union, overload
)

from more_itertools import zip_offset

from ._logging import logger
from ._metadata import __version__
from .colourspace import ASSColor, Opacity
from .convert import ConvertTime
from .exception import LineNotFoundWarning, MatchNotFoundError
from .font import Font, get_font
from .shape import Pixel, Shape
from .types import AnyPath, AssBool, AutoSlots, NamedMutableSequence, OrderedSet

_AssTextT = TypeVar('_AssTextT', bound='_AssText')


class Ass(AutoSlots):
    """Initialisation class containing all the information about an ASS file"""

    _meta: Meta
    _styles: List[Style]
    _lines: PList[Line]
    _output: Optional[AnyPath]
    _output_lines: List[str]
    _sections: Dict[str, _Section]
    _ptime: float
    _fix_timestamps: bool

    def __init__(
        self, input_: AnyPath, output: AnyPath | None = None, /,
        fps: Fraction | float = Fraction(24000, 1001),
        extended: bool = True, vertical_kanji: bool = False,
        fix_timestamps: bool = True
    ) -> None:
        """
        :param input_:              Input file path
        :param output:              Output file path
        :param fps:                 Framerate Per Second of the video related to the .ass file
        :param extended:            Add more info about the lines, words, syllables and chars for each line
        :param vertical_kanji:      Line text with alignment 4, 5 or 6 will be positioned vertically
                                    Additionally, ``line`` fields will be re-calculated based on the re-positioned ``line.chars``.
        :param fix_timestamps:      If True, will fix the timestamps on their real start and end time.
                                    If False, start and end times will just be the raw timestamps.
        """
        self._ptime = time.time()

        with open(input_, 'r', encoding='utf-8-sig') as file:
            lines_file = file.read()

        self._output = output
        fps = fps if isinstance(fps, Fraction) else Fraction(fps)
        self._output_lines = []

        # Find section pattern
        self._sections = {
            m.group(0): _Section(m.group(0), *m.span(0))
            for m in re.finditer(r'(^\[[^\]]*])', lines_file, re.MULTILINE)
        }

        # Slice text
        for sec1, sec2 in zip_offset(
            self._sections.values(), self._sections.values(), offsets=(0, 1),
            longest=True, fillvalue=_Section(start=None)
        ):
            sec1.text = lines_file[sec1.end:sec2.start]

        # Make a Meta object from both Script Info and Aegisub Project Garbage
        self._meta = Meta.from_text(
            ''.join(
                s.text for s in self._sections.values()
                if s.name in {'[Script Info]', '[Aegisub Project Garbage]'}),
            fps
        )
        # Make styles based on each line inside the [V4+ Styles] section
        # We remove the first line who starts by "Format:"
        self._styles = [
            Style.from_text(stext)
            for s in self._sections.values()
            if s.name == '[V4+ Styles]'
            for stext in s.text.strip().splitlines()[1:]
        ]
        # Extract basic data from each line
        # We also remove the first line who starts by "Format:"
        self._lines = PList(
            Line.from_text(ltext, i, fps, self._meta, self._styles, fix_timestamps)
            for s in self._sections.values()
            if s.name == '[Events]'
            for i, ltext in enumerate(s.text.strip().splitlines()[1:])
        )
        self._fix_timestamps = fix_timestamps

        if not extended:
            return None

        # Keep styles and lines linked to them for compute the leadin and leadout
        lines_by_styles: DefaultDict[str, List[Line]] = defaultdict(list)
        for line in self._lines:
            try:
                line_style = line.style
            except AttributeError:
                logger.user_warning(f'{LineNotFoundWarning()}: Line {line.i} is using an undefined style, skipping...')
                logger.debug(f'{line.i}: {line.raw_text}')
                continue
            lines_by_styles[line_style.name].append(line)

            # get_font uses lru_cache
            # It really boosts performance
            font = get_font(line_style)

            line.add_data(font)
            line.add_words(font)
            line.add_syls(font, vertical_kanji)
            line.add_chars(font, vertical_kanji)

        # Add durations between dialogs
        default_lead = 1 / float(fps) * round(fps)

        for preline, curline, postline in (
            zline
            for liness in lines_by_styles.values()
            for zline in zip_offset(liness, liness, liness, offsets=(-1, 0, 1), longest=True)
        ):
            if not curline:
                continue
            curline.leadin = default_lead if not preline else curline.start_time - preline.end_time
            curline.leadout = default_lead if not postline else postline.start_time - curline.end_time

    def __del__(self) -> None:
        logger.debug('Entering __del__ Ass...')
        get_font.cache_clear()
        logger.debug('Clear cache done!')

    @property
    def data(self) -> Tuple[Meta, List[Style], PList[Line]]:
        """
        :return:            Return data of the .ass file
        """
        return self._meta, self._styles, self._lines

    @property
    def meta(self) -> Meta:
        """
        :return:            Return the Meta of the .ass file
        """
        return self._meta

    @property
    def styles(self) -> List[Style]:
        """
        :return:            Return the list of styles included in the .ass file
        """
        return self._styles

    @property
    def lines(self) -> PList[Line]:
        """
        :return:            Return the list of lines included in the .ass file
        """
        return self._lines

    def add_line(self, line: Line, fix_timestamps: Optional[bool] = None) -> None:
        """
        Format a Line to a string suitable for writing into ASS file
        and add it to an internal list

        :param line:                Line object
        :param fix_timestamps:      If True, will fix the timestamps on their real start and end time.
                                    If False, start and end times will just be the raw timestamps.
        """
        self._output_lines.append(
            line.compose_ass_line(
                fix_timestamps=fix_timestamps if fix_timestamps is not None else self._fix_timestamps
            )
        )

    @logger.catch
    def save(self, lines: Optional[Iterable[Line]] = None, comment_original: bool = True, fix_timestamps: bool = True) -> None:
        """
        Write the lines added by :py:func:`add_line` to the output file specified in the constructor

        :param lines:               Additional Line objects to be written
        :param comment_original:    If True, will comment the original lines
        :param fix_timestamps:      If True, will fix the timestamps of the additional lines on their real start and end time.
                                    If False, start and end times will just be the raw timestamps.
        """
        if not self._output:
            raise ValueError('path_output hasn\'t been specified in the constructor')

        with open(self._output, 'w', encoding='utf-8-sig') as f:
            # Write header
            try:
                txt = re.sub(r'^; .*\r?\n', '', self._sections['[Script Info]'].text, 0, re.MULTILINE)
            except KeyError:
                txt = ''

            header_pyon = f'; Script generated by Pyonfx {__version__}\n; https://github.com/Ichunjo/PyonFX\n'
            f.write('[Script Info]\n' + header_pyon + txt.strip() + '\n\n')

            # Write garbage and styles
            for name in ('[Aegisub Project Garbage]', '[V4+ Styles]'):
                try:
                    section = self._sections[name]
                except KeyError:
                    pass
                else:
                    f.write(name + '\n' + section.text.strip() + '\n\n')

            # Write lines
            try:
                events_txt = self._sections['[Events]'].text.strip()
                f.write('[Events]\n')
                if comment_original:
                    f.write(re.sub(r'^Dialogue:|Comment:', 'Comment:', events_txt, 0, re.MULTILINE))
                else:
                    f.write('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')
            except KeyError:
                pass

            f.writelines(self._output_lines)
            if lines:
                f.writelines(
                    line.compose_ass_line(fix_timestamps=fix_timestamps if fix_timestamps is not None else self._fix_timestamps)
                    for line in lines
                )
            f.write('\n\n')

            # Extra data
            if '[Aegisub Extradata]' in self._sections:
                f.write(
                    '[Aegisub Extradata]\n'
                    + self._sections['[Aegisub Extradata]'].text.strip()
                    + '\n'
                )

        logger.user_info(f"Produced lines: {len(self._output_lines)}")
        logger.user_info(f"Process duration (in seconds): {round(time.time() - self._ptime, ndigits=3)}")

    @logger.catch
    def open_aegisub(self) -> None:
        """
        Open the output specified in the constructor with Aegisub.
        """
        if not self._output:
            raise ValueError('path_output hasn\'t been specified in the constructor')
        # Check if it was saved
        if sys.platform == 'win32':
            os.startfile(self._output)
        else:
            subprocess.call(['aegisub', self._output])

    @logger.catch
    def open_mpv(self, video_path: AnyPath | None = None, video_start: Optional[str] = None, full_screen: bool = False) -> None:
        """
        Open the output specified in the constructor with MPV.
        Please add MPV in your PATH (https://pyonfx.readthedocs.io/en/latest/quick%20start.html#installation-extra-step)

        :param video_path:          Video path. If not specified, will use the path in meta.video_file
        :param video_start:         Start time for the video (more info: https://mpv.io/manual/master/#options-start)
                                    If not specified, 0 is automatically taken
        :param full_screen:         Run MPV in full screen, defaults to False
        """
        if not self._output:
            raise ValueError('path_output hasn\'t been specified in the constructor')

        # Check if mpv is usable
        if self.meta.video_file.startswith('?dummy') and not video_path:
            raise FileNotFoundError(
                f'{self.__class__.__name__}: Cannot use MPV; dummy video detected'
            )

        # Setting up the command to execute
        cmd = ['mpv']

        if video_path:
            cmd.append(str(video_path))
        else:
            cmd.append(self.meta.video_file)
        if video_start:
            cmd.append('--start=' + video_start)
        if full_screen:
            cmd.append('--fs')

        cmd.append('--sub-file=' + str(self._output))

        try:
            subprocess.call(cmd)
        except FileNotFoundError as file_err:
            raise FileNotFoundError(f'{self.__class__.__name__}: MPV not found') from file_err


class _DataCore(AutoSlots, Iterable[Tuple[str, Any]], ABC, empty_slots=True):
    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for name in self.__slots__:
            try:
                yield name, getattr(self, name)
            except AttributeError:
                pass

    def __str__(self) -> str:
        try:
            return self._pretty_print(self)
        finally:
            self._pretty_print.cache_clear()

    def __repr__(self) -> str:
        return pformat(self._asdict())

    def _asdict(self) -> Dict[str, Any]:
        return {k: v._asdict() if isinstance(v, _DataCore) else v for k, v in self}

    @lru_cache(maxsize=None)
    def _pretty_print(self, obj: _DataCore, indent: int = 0, name: Optional[str] = None) -> str:
        if not name:
            out = " " * indent + f'{obj.__class__.__name__}:\n'
        else:
            out = " " * indent + f'{name}: ({obj.__class__.__name__}):\n'

        indent += 4
        for k, v in obj:
            if isinstance(v, _DataCore):
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


class Meta(_DataCore):
    """
    Meta object contains informations about the Ass.

    More info about each of them can be found on http://docs.aegisub.org/manual/Styles
    """
    play_res_x: int
    """Video width"""
    play_res_y: int
    """Video height"""

    title: str
    original_script: str
    original_translation: str
    original_editing: str
    original_timing: str
    synch_point: str
    script_updated_by: str
    update_details: str

    wrap_style: int
    """Determines how line breaking is applied to the subtitle line"""
    scaled_border_and_shadow: AssBool
    """Determines if it has to be used script resolution (*True*) or video resolution (*False*) to scale border and shadow"""

    y_cb_cr_matrix: str
    """YUV Matrix"""

    audio_file: str
    """Loaded audio path (absolute)"""
    video_file: str
    """Loaded video path (absolute)"""

    fps: Fraction
    """FrameRate per Second"""

    @classmethod
    def from_text(cls, text: str, fps: Fraction) -> Meta:
        """
        Make a Meta object from a chunk of text [Script Info] and [Aegisub Project Garbage]

        :param text:        Meta text
        :param fps:         Framerate Per Second
        :return:            Meta object
        """
        self = cls()
        # CamelCase to snake_case
        pattern = re.compile(r'(?<!^)(?=[A-Z])')

        for k, v in (
            (m.groupdict()['name'], m.groupdict()['value'])
            for m in re.finditer(r'^(?P<name>.*): (?P<value>.*)$', text, re.MULTILINE)
        ):
            k = pattern.sub('_', k.replace(' ', '')).lower()
            if k in self.__slots__:
                setattr(self, k, eval(cls.__annotations__[k])(v))
        self.fps = fps
        return self


class Style(_DataCore):
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
    @logger.catch(force_exit=True)
    def from_text(cls, text: str) -> Style:
        """
        Make a Style object from a .ass text line

        :param text:        Style text
        :return:            Style object
        """
        self = cls()

        if not (style_match := re.match(r'Style: (.+?)$', text)):
            raise MatchNotFoundError(f'{self.__class__.__name__}: No Style match found for this line!')
        # Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,
        # Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,
        # BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        style = style_match[1].split(',')

        self.name = str(style[0])

        self.fontname = str(style[1])
        self.fontsize = float(style[2])

        self.color1 = ASSColor(f'&H{style[3][4:]}&')
        self.color2 = ASSColor(f'&H{style[4][4:]}&')
        self.color3 = ASSColor(f'&H{style[5][4:]}&')
        self.color4 = ASSColor(f'&H{style[6][4:]}&')

        self.alpha1 = Opacity.from_ass_val(f'{style[3][:4]}&')
        self.alpha2 = Opacity.from_ass_val(f'{style[4][:4]}&')
        self.alpha3 = Opacity.from_ass_val(f'{style[5][:4]}&')
        self.alpha4 = Opacity.from_ass_val(f'{style[6][:4]}&')

        self.bold = style[7] == '-1'
        self.italic = style[8] == '-1'
        self.underline = style[9] == '-1'
        self.strikeout = style[10] == '-1'

        self.scale_x = float(style[11])
        self.scale_y = float(style[12])

        self.spacing = float(style[13])
        self.angle = float(style[14])

        self.border_style = style[15] == '3'
        self.outline = float(style[16])
        self.shadow = float(style[17])

        self.alignment = int(style[18])
        self.margin_l = int(style[19])
        self.margin_r = int(style[20])
        self.margin_v = int(style[21])

        self.encoding = int(style[22])

        return self


class _PositionedText(_DataCore, ABC, empty_slots=True):
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
        for k, v in self:
            setattr(obj, k, v)
        return obj

    def __deepcopy__(self: _AssTextT, *args: Any) -> _AssTextT:
        obj = self.__class__()
        for k, v in self:
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

    def to_shape(self, fscx: Optional[float] = None, fscy: Optional[float] = None, copy: bool = True) -> Shape:
        """
        Convert current AssText object to shape based on its Style attribute.

        ::

            l = line.deep_copy()
            l.text = f"{\\\\an7\\\\pos({line.left},{line.top})\\\\p1}{line.to_shape()}"
            io.write_line(l)

        :param fscx:        The scale_x value for the shape, default to current scale_x object
        :param fscy:        The scale_y value for the shape, default to current scale_y object
        :return:            Shape object, representing the text
        """
        # Obtaining information and editing values of style if requested
        if copy:
            obj = self.deep_copy()
        else:
            obj = self

        # Editing temporary the style to properly get the shape
        if fscx is not None:
            obj.style.scale_x = fscx
        if fscy is not None:
            obj.style.scale_y = fscy

        # Obtaining font information from style and obtaining shape
        font = get_font(obj.style)
        shape = font.text_to_shape(obj.text)
        # Clearing resources to not let overflow errors take over
        del font, obj

        return shape

    def to_clip(self, an: int = 7, fscx: Optional[float] = None, fscy: Optional[float] = None, copy: bool = True) -> Shape:
        """
        Convert current AssText object to shape based on its Style attribute, suitable for \\clip tag

        ::

            l = line.deep_copy()
            l.text = f"{\\\\an5\\\\pos({line.center},{line.middle})\\\\clip({line.to_clip()})}{line.text}"
            io.write_line(l)

        :param an:          Alignment wanted for the shape
        :param fscx:        The scale_x value for the shape, default to current scale_x object
        :param fscy:        The scale_y value for the shape, default to current scale_y object
        :return:            A Shape object, representing the text with the style format values of the object
        """
        if copy:
            obj = self.deep_copy()
        else:
            obj = self

        # Setting default values
        if fscx is None:
            fscx = obj.style.scale_x
        if fscy is None:
            fscy = obj.style.scale_y

        # Obtaining text converted to shape
        shape = obj.to_shape(fscx, fscy, False)
        shape.align(an)

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
    @logger.catch(force_exit=True)
    def from_text(cls, text: str, i: int, fps: Fraction,
                  meta: Optional[Meta] = None, styles: Optional[Iterable[Style]] = None,
                  fix_timestamps: bool = True) -> Line:
        """
        Make a Line object from a .ass text line

        :param text:            An .ass line starting by "Dialogue" or "Comment"
        :param i:               Line index
        :param fps:             FrameRate Per Second
        :param meta:            Meta object to link to the Line
        :param styles:          Iterable of Style, defaults to None
        :param fix_timestamps:  If True, will fix the timestamps on their real start and end time.
                                If False, start and end times will just be the raw timestamps.
        :return:            A Line object
        """
        self = cls()

        # Analysing line
        if not (anal_line := re.match(r"(Dialogue|Comment): (.+?)$", text)):
            raise MatchNotFoundError(f'{self.__class__.__name__}: No Line match found for this line!')
        # Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        self.i = i

        self.comment = anal_line[1] == "Comment"
        linesplit = anal_line[2].split(",")

        self.layer = int(linesplit[0])

        if fix_timestamps:
            self.start_time = ConvertTime.assts2seconds(linesplit[1], fps, is_start=True)
            self.end_time = ConvertTime.assts2seconds(linesplit[2], fps, is_start=False)
        else:
            self.start_time = ConvertTime.ts2seconds(linesplit[1])
            self.end_time = ConvertTime.ts2seconds(linesplit[2])
        self.duration = self.end_time - self.start_time

        if styles:
            for style in styles:
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
        """
        Add more data to the current object based on given Font object

        :param font:        Font object
        """
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
        """
        Add data on words based on a Font object

        :param font:        Font object
        """
        # Adding words
        self.words = PList()

        for wi, mmatch in enumerate(re.finditer(r"(\s*)([^\s]+)(\s*)", self.text)):
            prespace, word_text, postspace = mmatch.groups()
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
        space_width = font.text_extents(' ').width

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
            max_width = max(word.width for word in self.words)
            sum_height = sum((word.height for word in self.words), 0.)

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
        """
        Add data on syllables based on a Font object

        :param font:            Font object
        :param vertical_kanji:  Line text with alignment 4, 5 or 6 will be positioned vertically
        """
        self.syls = PList()

        syldata = re.compile(r'{(?P<pretags>.*?)\\[kK][of]?(?P<kdur>\d+)(?P<posttags>[^}]*)}(?P<syltext>[^{]*)')
        slash = re.compile(r'\\\\')
        ppsyl = re.compile(r'(\s*).*?(\s*)$')

        ks = tuple(syldata.finditer(self.raw_text.replace('}{', '').replace('\\k', '}{\\k').replace('{}', '')))

        last_time = 0.0
        word_i = 0
        for si, (k0, k1) in enumerate(zip_offset(ks, ks, offsets=(0, 1), longest=True)):
            assert k0
            syl = Syllable()
            # Indices
            syl.i = si
            syl.word_i = word_i

            syl.text = k0.groupdict()['syltext']
            if not syl.text or syl.text.isspace():
                syl.prespace, syl.postspace = 0, 0
            elif ppsp := ppsyl.match(syl.text):
                syl.prespace, syl.postspace = (len(x) for x in ppsp.groups())
            syl.width, syl.height = font.text_extents(syl.text)
            syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font.metrics

            if (
                syl.text.endswith(' ')
            ) or (
                k1 and k1.groupdict()['syltext'].startswith(' ')
            ):
                word_i += 1

            syl.start_time = last_time
            # kdur is in centiseconds
            # Converting in seconds...
            kdur = k0.groupdict()['kdur']
            syl.end_time = last_time + int(kdur) / 100
            syl.duration = int(kdur) / 100

            last_time = syl.end_time

            for ptag in (
                ptag
                for tagspos in ('pretags', 'posttags')
                for ptag in slash.split(k0.groupdict()[tagspos].replace('}{', ''))
            ):
                if ptag.startswith('\\-'):
                    if hasattr(syl, 'inline_fx'):
                        syl.inline_fx.add(ptag.strip('\\-'))
                    else:
                        syl.inline_fx = OrderedSet([ptag.strip('\\-')])
                elif ptag:
                    if hasattr(syl, 'tags'):
                        syl.tags.add(ptag)
                    else:
                        syl.tags = OrderedSet([ptag])

            self.syls.append(syl)

        try:
            style = self.style
            meta = self.meta
        except AttributeError:
            return None

        space_width = font.text_extents(" ").width

        if style.an_is_top() or style.an_is_bottom() or not vertical_kanji:
            cur_x = self.left
            for syl in self.syls:
                syl.style = style
                syl.meta = meta

                cur_x += syl.prespace * (space_width + style.spacing)

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

                cur_x += syl.width + syl.postspace * (space_width + style.spacing) + style.spacing

                # Vertical position
                syl.top = self.top
                syl.middle = self.middle
                syl.bottom = self.bottom
                syl.y = self.y
            return None

        # Kanji vertical position
        if vertical_kanji:
            max_width = max(syl.width for syl in self.syls)
            sum_height = sum((syl.height for syl in self.syls), 0.)

            cur_y = meta.play_res_y / 2 - sum_height / 2

            for syl in self.syls:
                syl.style = style
                syl.meta = meta
                # Horizontal position
                x_fix = (max_width - syl.width) / 2
                syl.style = style
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
        """
        Add data on chars based on a Font object

        :param font:            Font object
        :param vertical_kanji:  Line text with alignment 4, 5 or 6 will be positioned vertically
        """
        # Adding chars
        self.chars = PList()

        # If we have syls in line, we prefert to work with them to provide more informations
        if not self.syls and not self.words:
            return None
        words_or_syls: Union['PList[Syllable]', 'PList[Word]'] = self.syls if self.syls else self.words

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
            meta, style = self.meta, self.style
        except AttributeError:
            return None

        if style.an_is_top() or style.an_is_bottom() or not vertical_kanji:
            cur_x = self.left
            for char in self.chars:
                # Horizontal position
                char.left = cur_x
                char.center = char.left + char.width / 2
                char.right = char.left + char.width

                if style.an_is_left():
                    char.x = char.left
                if style.an_is_center():
                    char.x = char.center
                else:
                    char.x = char.right

                cur_x += char.width + style.spacing

                # Vertical position
                char.top = self.top
                char.middle = self.middle
                char.bottom = self.bottom
                char.y = self.y
        else:
            max_width = max(char.width for char in self.chars)
            sum_height = sum((char.height for char in self.chars), 0.)

            cur_y = x_fix = meta.play_res_y / 2 - sum_height / 2

            # Fixing line positions
            self.top = cur_y
            self.middle = meta.play_res_y / 2
            self.bottom = self.top + sum_height
            self.width = max_width
            self.height = sum_height
            if style.alignment == 4:
                self.center = self.left + max_width / 2
                self.right = self.left + max_width
            elif style.alignment == 5:
                self.left = self.center - max_width / 2
                self.right = self.left + max_width
            else:
                self.left = self.right - max_width
                self.center = self.left + max_width / 2

            for char in self.chars:
                # Horizontal position
                x_fix = (max_width - char.width) / 2
                if style.alignment == 4:
                    char.left = self.left + x_fix
                    char.center = char.left + char.width / 2
                    char.right = char.left + char.width
                    char.x = char.left
                elif style.alignment == 5:
                    char.left = meta.play_res_x / 2 - char.width / 2
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

    def compose_ass_line(self, *, fix_timestamps: bool = True) -> str:
        """
        Make an ASS line suitable for writing into an .ass file

        :param fix_timestamps:      If True, will fix the timestamps on their real start and end time.
                                    If False, start and end times will just be the raw timestamps.
        """
        ass_line = 'Comment: ' if self.comment else 'Dialogue: '
        if fix_timestamps:
            start = ConvertTime.seconds2assts(self.start_time, self.meta.fps, is_start=True)
            end = ConvertTime.seconds2assts(self.end_time, self.meta.fps, is_start=False)
        else:
            start, end = [ConvertTime.seconds2ts(t, precision=3)[1:-1] for t in [self.start_time, self.end_time]]
        elements: List[str] = [
            str(self.layer),
            start, end,
            self.style.name, self.actor,
            str(self.margin_l), str(self.margin_r), str(self.margin_v),
            self.effect, self.text
        ]
        return ass_line + ','.join(elements) + '\n'


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
    inline_fx: OrderedSet[str]
    """Inline effect (marked as \\-EFFECT in karaoke-time)"""


class Syllable(_WordElement):
    """
    Syllable object contains informations about a single syl of a line in the Ass.

    A syl can be defined as some text after a karaoke tag (k, ko, kf)
    (e.g.: In ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, "Pyon" and "FX" are distinct syllables),
    """
    tags: OrderedSet[str]
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


if TYPE_CHECKING:
    BaseUserList = UserList
else:
    if sys.version_info < (3, 9):
        class _UserList(UserList):
            def __getitem__(self, item: Any) -> Any:
                if isinstance(item, (slice, int)):
                    return super().__getitem__(item)
                return self.__class__

        BaseUserList = _UserList()
    else:
        BaseUserList = UserList


class PList(BaseUserList[_AssTextT]):
    """PyonFX list"""

    def __init__(self, __iterable: Iterable[_AssTextT] | None = None, /) -> None:
        """
        If no argument is given, the constructor creates a new empty list.

        :param iterable:            Iterable object, defaults to None
        """
        super().__init__(__iterable)

    def __str__(self) -> str:
        return '\n'.join(str(at) for at in self)

    def __repr__(self) -> str:
        return '\n'.join(repr(at) for at in self)

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


class _Section(NamedMutableSequence[Any]):
    name: str
    start: int
    end: Optional[int]
    text: str
