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

import copy
import os
import re
import subprocess
import sys
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast

from .convert import Convert
from .font_utility import Font

__all__ = [
    'Meta', 'Style',
    'Line', 'Word', 'Syllable', 'Char',
    'Ass'
]


class DataCore(ABC):
    """Abstract DataCore object"""

    def __str__(self) -> str:
        return self._pretty_print(self)

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
            elif isinstance(v, Color):
                out += " " * indent + f"{k}: {v.pprint()}\n"
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
    audio: Union[Path, str]
    """Loaded audio path (absolute)"""
    video: Union[Path, str]
    """Loaded video path (absolute)"""


class Color(str):
    alpha: str

    def __new__(cls, x: str, alpha: str) -> Color:
        color = super().__new__(cls, x)
        color.alpha = alpha
        return color

    def pprint(self) -> str:
        return super().__str__() + f' | Alpha: {self.alpha}'


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
    color1: Color
    """Primary color (fill) and transparency"""
    color2: Color
    """Secondary color (secondary fill, for karaoke effect) and transparency"""
    color3: Color
    """Outline (border) color and transparency"""
    color4: Color
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
    def alignment(self, an: int) -> None:
        if 1 <= an <= 9:
            self._alignment = an
        else:
            raise ValueError('Alignment of the text must be <= 9 or >= 1')

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
    start_time: int
    """Start time (in milliseconds)"""
    end_time: int
    """End time (in milliseconds)"""
    duration: int
    """Duration (in milliseconds)"""
    text: str
    """Text"""
    style: Style
    """Reference to the Style object"""
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

    @abstractmethod
    def deep_copy(self) -> AssText:
        """
        Returns:
            A deep copy of this object
        """
        return copy.deepcopy(self)


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
    """Time between this line and the previous one (in milliseconds; first line = 1000.1) (*)"""
    leadout: float
    """Time between this line and the next one (in milliseconds; first line = 1000.1) (*)"""
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

    def deep_copy(self) -> Line:
        return cast(Line, super().deep_copy())

    def compose_ass_line(self) -> str:
        return (
            "Comment" if self.comment else "Dialogue"
            + str(self.layer)
            + str(Convert.time(max(0, int(self.start_time))))
            + str(Convert.time(max(0, int(self.end_time))))
            + self.style.name + self.actor
            + str(self.margin_l) + str(self.margin_r) + str(self.margin_v)
            + self.effect + self.text
        )


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

    def deep_copy(self) -> Word:
        return cast(Word, super().deep_copy())


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

    def deep_copy(self) -> Syllable:
        return cast(Syllable, super().deep_copy())


class Char(WordElement):
    """
    Char object contains informations about a single char of a line in the Ass.

    A char is defined by some text between two karaoke tags (k, ko, kf).
    """
    syl_i: int
    """Char syl index (e.g.: In line text ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, letter "F" will have syl_i=3)"""
    syl_char_i: int
    """Char invidual syl index (e.g.: In line text ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, letter "e" of "users" will have syl_char_i=2)"""

    def deep_copy(self) -> Char:
        return cast(Char, super().deep_copy())



class Ass:
    """Initialization class containing all the information about an ASS file"""
    path_input: Path
    path_output: Path
    vertical_kanji: bool

    meta: Meta
    styles: List[Style]
    lines: List[Line]

    _output: List[str]
    _output_lines: List[str] = []
    _output_extradata: List[str]

    _ptime: float

    class _TextChunk(NamedTuple):
        tags: str
        text: str
        word_i: Optional[int] = None

    def __init__(self, path_input: Union[os.PathLike[str], str], path_output: Union[os.PathLike[str], str],
                 comment_original: bool = True, extended: bool = True, vertical_kanji: bool = False) -> None:
        """
        Args:
            path_input (str, optional):
                Input file path

            path_output (str, optional):
                Output file path

            comment_original (bool, optional):
                If True, you will find all the lines of the input file commented before the new lines generated.
                Defaults to True.

            extended (bool, optional):
                Calculate more informations from lines.
                Defaults to True.

            vertical_kanji (bool, optional):
                If True, line text with alignment 4, 5 or 6 will be positioned vertically.
                Additionally, ``line`` fields will be re-calculated based on the re-positioned ``line.chars``.
                Defaults to False.
        """
        # Starting to take process time
        self._ptime = time.time()

        # Resolve paths, make absolute the paths
        # script_path = Path(sys.argv[0]).resolve()

        self.path_input = Path(path_input).resolve()
        self.path_output = Path(path_output).resolve()
        self.vertical_kanji = vertical_kanji
        self._output = []
        self._output_extradata = []

        self.meta = Meta()
        self.styles = []
        self.lines = []

        # Checking sub file validity (does it exists?)
        if not self.path_input.exists():
            raise FileNotFoundError(f"{path_input} is not a valid path!")

        with open(self.path_input, "r", encoding="utf-8-sig") as file:
            lines_file = file.readlines()

        section = ""
        nbli = 0
        for line in lines_file:
            # Vardë NOTE: I don't get that
            # Getting section
            section_pattern = re.compile(r"^\[([^\]]*)")
            if section_pattern.match(line):
                # Updating section
                section = section_pattern.match(line)[1]  # type: ignore
                # Appending line to output
                if section != "Aegisub Extradata":
                    self._output.append(line)
            elif section in {"Script Info", "Aegisub Project Garbage"}:
                self._parse_meta_data(line)
            elif section == "V4+ Styles":
                self._parse_style(line)
            elif section == "Events":
                self._parse_dialogue(line, nbli, comment_original)
                nbli += 1
            elif section == "Aegisub Extradata":
                self._output_extradata.append(line)
            else:
                raise ValueError(f"Unexpected section in the input file: [{section}]")

        # Adding informations to lines and meta?
        if extended:
            lines_by_styles: Dict[Style, List[Line]] = {}
            for line in self.lines:
                # Append dialog to styles (for leadin and leadout later)
                lines_by_styles.setdefault(line.style, [])
                lines_by_styles[line.style].append(line)

                line.duration = line.end_time - line.start_time
                line.text = re.sub(r"\{.*?\}", "", line.raw_text)

                # Add dialog text sizes and positions (if possible)
                if hasattr(line, 'style'):
                    font = Font(line.style)
                    font_metrics = font.get_metrics()

                    # Add line data
                    line = self._add_data_line(line, font, font_metrics)

                    # Calculating space width and saving spacing
                    space_width = font.get_text_extents(" ")[0]
                    style_spacing = line.style.spacing

                    # Add words data
                    line = self._add_data_words(line, font, font_metrics, space_width, style_spacing)

                    # Search for dialog's text chunks, to later create syllables
                    # A text chunk is a text with one or more {tags} preceding it
                    # Tags can be some text or empty string
                    text_chunks = self._search_text_chunks(line)

                    # Adding syls
                    line = self._add_data_syls(line, font, font_metrics, text_chunks, space_width, style_spacing)

                    # Getting chars
                    line = self._add_data_chars(line, font, font_metrics, style_spacing)
                else:
                    warnings.warn(f'Line {line.i} is using undefined style, skipping...', Warning)

            # Add durations between dialogs
            for _, liness in lines_by_styles.items():
                liness.sort(key=lambda x: x.start_time)
                for li, line in enumerate(liness):
                    line.leadin = (
                        1000.1
                        if li == 0
                        else line.start_time - liness[li - 1].end_time
                    )
                    line.leadout = (
                        1000.1
                        if li == len(liness) - 1
                        else liness[li + 1].start_time - line.end_time
                    )

    def _parse_meta_data(self, line: str) -> None:
        # Switch
        if valm := re.match(r"WrapStyle: *?(\d+)$", line):
            self.meta.wrap_style = int(valm[1].strip())
        elif valm := re.match(r"ScaledBorderAndShadow: *?(.+)$", line):
            self.meta.scaled_border_and_shadow = valm[1].strip() == "yes"
        elif valm := re.match(r"PlayResX: *?(\d+)$", line):
            self.meta.play_res_x = int(valm[1].strip())
        elif valm := re.match(r"PlayResY: *?(\d+)$", line):
            self.meta.play_res_y = int(valm[1].strip())
        elif valm := re.match(r"Audio File: *?(.*)$", line):
            if path := self._get_media_abs_path(valm[1].strip()):
                self.meta.audio = path
                line = f"Audio File: {str(self.meta.audio)}\n"
        elif valm := re.match(r"Video File: *?(.*)$", line):
            if path := self._get_media_abs_path(valm[1].strip()):
                self.meta.video = path
                line = f"Video File: {str(self.meta.video)}\n"
        # Appending line to output
        self._output.append(line)

    def _parse_style(self, line: str) -> None:
        self._output.append(line)
        style = re.match(r"Style: (.+?)$", line)

        if style:
            # Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,
            # Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,
            # BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
            style = style[1].split(",")
            nstyle = Style()

            nstyle.name = str(style[0])

            nstyle.fontname = str(style[1])
            nstyle.fontsize = float(style[2])

            nstyle.color1 = Color(f"&H{style[3][4:]}&", f"{style[3][:4]}&")
            nstyle.color2 = Color(f"&H{style[4][4:]}&", f"{style[4][:4]}&")
            nstyle.color3 = Color(f"&H{style[5][4:]}&", f"{style[5][:4]}&")
            nstyle.color4 = Color(f"&H{style[6][4:]}&", f"{style[6][:4]}&")

            nstyle.bold = style[7] == "-1"
            nstyle.italic = style[8] == "-1"
            nstyle.underline = style[9] == "-1"
            nstyle.strikeout = style[10] == "-1"

            nstyle.scale_x = float(style[11])
            nstyle.scale_y = float(style[12])

            nstyle.spacing = float(style[13])
            nstyle.angle = float(style[14])

            nstyle.border_style = style[15] == "3"
            nstyle.outline = float(style[16])
            nstyle.shadow = float(style[17])

            nstyle.alignment = int(style[18])
            nstyle.margin_l = int(style[19])
            nstyle.margin_r = int(style[20])
            nstyle.margin_v = int(style[21])

            nstyle.encoding = int(style[22])

            self.styles.append(nstyle)

    def _parse_dialogue(self, line: str, nb: int, comment_original: bool) -> None:
        # Appending line to output (commented) if comment_original is True
        if comment_original:
            self._output.append(
                re.sub(r"^(Dialogue|Comment):", "Comment:", line, count=1)
            )
        elif line.startswith("Format"):
            self._output.append(line.strip())

        # Analysing line
        if anal_line := re.match(r"(Dialogue|Comment): (.+?)$", line):
            # Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            nline = Line()

            nline.i = nb

            nline.comment = anal_line[1] == "Comment"
            linesplit = anal_line[2].split(",")

            nline.layer = int(linesplit[0])

            nline.start_time = Convert.time(linesplit[1])
            nline.end_time = Convert.time(linesplit[2])

            for style in self.styles:
                if style.name == linesplit[3]:
                    nline.style = style
                    break
            nline.actor = linesplit[4]

            nline.margin_l = int(linesplit[5])
            nline.margin_r = int(linesplit[6])
            nline.margin_v = int(linesplit[7])

            nline.effect = linesplit[8]

            nline.raw_text = ",".join(linesplit[9:])

            self.lines.append(nline)

    def _get_media_abs_path(self, mediafile: str) -> Union[Path, str]:
        """
        Internal function that tries to get the absolute path for media files in meta
        If this is not a dummy video, let's try to get the absolute path for the video
        """
        return Path(mediafile).resolve() if not mediafile.startswith("?dummy") else mediafile

    def _add_data_line(self, line: Line, font: Font, font_metrics: Tuple[float, float, float, float]) -> Line:
        line.width, line.height = font.get_text_extents(line.text)
        line.ascent, line.descent, line.internal_leading, line.external_leading = font_metrics

        # If self.meta has play_res_x then we assume it has play_res_y too
        if hasattr(self.meta, 'play_res_x'):
            # Horizontal position
            margin_l = (
                line.margin_l if line.margin_l != 0 else line.style.margin_l
            )
            margin_r = (
                line.margin_r if line.margin_r != 0 else line.style.margin_r
            )
            if line.style.an_is_left():
                line.left = margin_l
                line.center = line.left + line.width / 2
                line.right = line.left + line.width
                line.x = line.left
            elif line.style.an_is_center():
                line.left = self.meta.play_res_x / 2 - line.width / 2 + margin_l / 2 - margin_r / 2
                line.center = line.left + line.width / 2
                line.right = line.left + line.width
                line.x = line.center
            else:
                line.left = self.meta.play_res_x - margin_r - line.width
                line.center = line.left + line.width / 2
                line.right = line.left + line.width
                line.x = line.right

            # Vertical position
            if line.style.an_is_top():
                line.top = line.margin_v if line.margin_v != 0 else line.style.margin_v
                line.middle = line.top + line.height / 2
                line.bottom = line.top + line.height
                line.y = line.top
            elif line.style.an_is_middle():
                line.top = self.meta.play_res_y / 2 - line.height / 2
                line.middle = line.top + line.height / 2
                line.bottom = line.top + line.height
                line.y = line.middle
            else:
                line.top = self.meta.play_res_y - (line.margin_v if line.margin_v != 0 else line.style.margin_v) - line.height
                line.middle = line.top + line.height / 2
                line.bottom = line.top + line.height
                line.y = line.bottom

        return line

    def _add_data_words(self, line: Line, font: Font, font_metrics: Tuple[float, float, float, float], space_width: float, style_spacing: float) -> Line:
        # Adding words
        line.words = []

        presp_txt_postsp = re.findall(r"(\s*)([^\s]+)(\s*)", line.text)
        presp_txt_postsp = cast(List[Tuple[str, str, str]], presp_txt_postsp)

        for wi, (prespace, word_text, postspace) in enumerate(presp_txt_postsp):
            word = Word()

            word.i = wi

            word.start_time = line.start_time
            word.end_time = line.end_time
            word.duration = line.duration

            word.style = line.style
            word.text = word_text

            word.prespace = len(prespace)
            word.postspace = len(postspace)

            word.width, word.height = font.get_text_extents(word.text)
            word.ascent, word.descent, word.internal_leading, word.external_leading = font_metrics

            line.words.append(word)

        # Calculate word positions with all words data already available
        # If self.meta has play_res_x then we assume it has play_res_y too
        if line.words and hasattr(self.meta, 'play_res_x'):
            if line.style.an_is_top() or line.style.an_is_bottom():
                cur_x = line.left
                for word in line.words:
                    # Horizontal position
                    cur_x += word.prespace * space_width + style_spacing
                    word.left = cur_x
                    word.center = word.left + word.width / 2
                    word.right = word.left + word.width

                    if line.style.an_is_left():
                        word.x = word.left
                    elif line.style.an_is_center():
                        word.x = word.center
                    else:
                        word.x = word.right

                    # Vertical position
                    word.top = line.top
                    word.middle = line.middle
                    word.bottom = line.bottom
                    word.y = line.y

                    # Updating cur_x
                    cur_x += word.width + word.postspace * (space_width + style_spacing) + style_spacing
            else:
                max_width, sum_height = 0, 0
                for word in line.words:
                    max_width = max(max_width, word.width)
                    sum_height = sum_height + word.height

                cur_y = x_fix = self.meta.play_res_y / 2 - sum_height / 2
                for word in line.words:
                    # Horizontal position
                    x_fix = (max_width - word.width) / 2

                    if line.style.alignment == 4:
                        word.left = line.left + x_fix
                        word.center = word.left + word.width / 2
                        word.right = word.left + word.width
                        word.x = word.left
                    elif line.style.alignment == 5:
                        word.left = self.meta.play_res_x / 2 - word.width / 2
                        word.center = word.left + word.width / 2
                        word.right = word.left + word.width
                        word.x = word.center
                    else:
                        word.left = line.right - word.width - x_fix
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

        return line

    def _search_text_chunks(self, line: Line) -> List['_TextChunk']:
        # Search for dialog's text chunks, to later create syllables
        # A text chunk is a text with one or more {tags} preceding it
        # Tags can be some text or empty string

        text_chunks = []
        tag_pattern = re.compile(r"(\{.*?\})+")
        tag = tag_pattern.search(line.raw_text)
        word_i = 0

        if not tag:
            # No tags found
            text_chunks.append(self._TextChunk('', line.raw_text))
        else:
            # First chunk without tags?
            if tag.start() != 0:
                text_chunks.append(
                    self._TextChunk('', line.raw_text[0:tag.start()])
                )

            # Searching for other tags
            while True:
                next_tag = tag_pattern.search(line.raw_text, tag.end())
                chk = self._TextChunk(
                    line.raw_text[tag.start() + 1 : tag.end() - 1].replace("}{", ""),
                    line.raw_text[tag.end() : (next_tag.start() if next_tag else None)],
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

    def _add_data_syls(self, line: Line, font: Font, font_metrics: Tuple[float, float, float, float],
                       text_chunks: List['_TextChunk'], space_width: float, style_spacing: float) -> Line:
        # Adding syls
        si = 0
        last_time = 0
        inline_fx = ""
        syl_tags_pattern = re.compile(r"(.*?)\\[kK][of]?(\d+)(.*)")

        line.syls = []
        for tc in text_chunks:
            # If we don't have at least one \k tag, everything is invalid
            if not syl_tags_pattern.match(tc.tags):
                line.syls.clear()
                break

            posttags = tc.tags
            syls_in_text_chunk: List[Syllable] = []
            while True:
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

                        syl.width, syl.height = font.get_text_extents("")
                        syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font_metrics

                        line.syls.append(syl)

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

                    syl.width, syl.height = font.get_text_extents(syl.text)
                    syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font_metrics

                    line.syls.append(syl)
                    break

                pretags, kdur, posttags = tags_syl.groups()

                # Create a Syllable object
                syl = Syllable()

                syl.start_time = last_time
                syl.end_time = last_time + int(kdur) * 10
                syl.duration = int(kdur) * 10

                syl.style = line.style
                syl.tags = pretags

                syl.i = si
                if tc.word_i is not None:
                    syl.word_i = tc.word_i

                syls_in_text_chunk.append(syl)

                # Update working variable
                si += 1
                last_time = syl.end_time

        # Calculate syllables positions with all syllables data already available
        if line.syls and hasattr(self.meta, 'play_res_x'):
            if line.style.an_is_top() or line.style.an_is_bottom() or not self.vertical_kanji:
                cur_x = line.left
                for syl in line.syls:
                    cur_x += syl.prespace * (space_width + style_spacing)

                    # Horizontal position
                    syl.left = cur_x
                    syl.center = syl.left + syl.width / 2
                    syl.right = syl.left + syl.width

                    if line.style.an_is_left():
                        syl.x = syl.left
                    elif line.style.an_is_center():
                        syl.x = syl.center
                    else:
                        syl.x = syl.right

                    cur_x += syl.width + syl.postspace * (space_width + style_spacing) + style_spacing

                    # Vertical position
                    syl.top = line.top
                    syl.middle = line.middle
                    syl.bottom = line.bottom
                    syl.y = line.y

            else:  # Kanji vertical position
                max_width, sum_height = 0, 0
                for syl in line.syls:
                    max_width = max(max_width, syl.width)
                    sum_height += syl.height

                cur_y = self.meta.play_res_y / 2 - sum_height / 2

                for syl in line.syls:
                    # Horizontal position
                    x_fix = (max_width - syl.width) / 2
                    if line.style.alignment == 4:
                        syl.left = line.left + x_fix
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.left
                    elif line.style.alignment == 5:
                        syl.left = line.center - syl.width / 2
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.center
                    else:
                        syl.left = line.right - syl.width - x_fix
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.right

                    # Vertical position
                    syl.top = cur_y
                    syl.middle = syl.top + syl.height / 2
                    syl.bottom = syl.top + syl.height
                    syl.y = syl.middle
                    cur_y += syl.height

        return line

    def _add_data_chars(self, line: Line, font: Font, font_metrics: Tuple[float, float, float, float], style_spacing: float) -> Line:
        # Adding chars
        line.chars = []

        # If we have syls in line, we prefert to work with them to provide more informations
        words_or_syls = line.syls if line.syls else line.words

        # Getting chars
        # char_index = 0
        for el in words_or_syls:
            el_text = "{}{}{}".format(" " * el.prespace, el.text, " " * el.postspace)
            for ci, char_text in enumerate(el_text):
                char = Char()
                char.i = ci

                # If we're working with syls, we can add some indexes
                if line.syls:
                    # el = cast(Syllable, el)
                    char.word_i = el.word_i  # type: ignore
                    char.syl_i = el.i
                    char.syl_char_i = ci
                else:
                    # el = cast(Word, el)
                    char.word_i = el.i

                # Adding last fields based on the existance of syls or not
                char.start_time = el.start_time
                char.end_time = el.end_time
                char.duration = el.duration

                char.style = line.style
                char.text = char_text

                char.width, char.height = font.get_text_extents(char.text)
                char.ascent, char.descent, char.internal_leading, char.external_leading = font_metrics

                line.chars.append(char)

        # Calculate character positions with all characters data already available
        if line.chars and hasattr(self.meta, 'play_res_x'):
            if line.style.an_is_top() or line.style.an_is_bottom() or not self.vertical_kanji:
                cur_x = line.left
                for char in line.chars:
                    # Horizontal position
                    char.left = cur_x
                    char.center = char.left + char.width / 2
                    char.right = char.left + char.width

                    if line.style.an_is_left():
                        char.x = char.left
                    if line.style.an_is_center():
                        char.x = char.center
                    else:
                        char.x = char.right

                    cur_x += char.width + style_spacing

                    # Vertical position
                    char.top = line.top
                    char.middle = line.middle
                    char.bottom = line.bottom
                    char.y = line.y
            else:
                max_width, sum_height = 0, 0
                for char in line.chars:
                    max_width = max(max_width, char.width)
                    sum_height = sum_height + char.height

                cur_y = x_fix = self.meta.play_res_y / 2 - sum_height / 2

                # Fixing line positions
                line.top = cur_y
                line.middle = self.meta.play_res_y / 2
                line.bottom = line.top + sum_height
                line.width = max_width
                line.height = sum_height
                if line.style.alignment == 4:
                    line.center = line.left + max_width / 2
                    line.right = line.left + max_width
                elif line.style.alignment == 5:
                    line.left = line.center - max_width / 2
                    line.right = line.left + max_width
                else:
                    line.left = line.right - max_width
                    line.center = line.left + max_width / 2

                for char in line.chars:
                    # Horizontal position
                    x_fix = (max_width - char.width) / 2
                    if line.style.alignment == 4:
                        char.left = line.left + x_fix
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.left
                    elif line.style.alignment == 5:
                        char.left = self.meta.play_res_x / 2 - char.width / 2
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.center
                    else:
                        char.left = line.right - char.width - x_fix
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.right

                    # Vertical position
                    char.top = cur_y
                    char.middle = char.top + char.height / 2
                    char.bottom = char.top + char.height
                    char.y = char.middle
                    cur_y += char.height

        return line

    def get_data(self) -> Tuple[Meta, List[Style], List[Line]]:
        """Utility function to easily retrieve meta, styles and lines.

        Returns:
            :attr:`meta`, :attr:`styles` and :attr:`lines`
        """
        return self.meta, self.styles, self.lines

    def write_line(self, line: Line) -> None:
        """Appends a line to the output list (which is private) that later on will be written to the output file when calling save().

        Use it whenever you've prepared a line, it will not impact performance since you
        will not actually write anything until :func:`save` will be called.

        Parameters:
            line (:class:`Line`): A line object. If not valid, TypeError is raised.
        """
        if isinstance(line, Line):
            self.__output.append(
                "\n%s: %d,%s,%s,%s,%s,%04d,%04d,%04d,%s,%s"
                % (
                    "Comment" if line.comment else "Dialogue",
                    line.layer,
                    Convert.time(max(0, int(line.start_time))),
                    Convert.time(max(0, int(line.end_time))),
                    line.style,
                    line.actor,
                    line.margin_l,
                    line.margin_r,
                    line.margin_v,
                    line.effect,
                    line.text,
                )
            )
            self.__plines += 1
        else:
            raise TypeError("Expected Line object, got %s." % type(line))

    def save(self, quiet: bool = False) -> None:
        """Write everything inside the private output list to a file.

        Parameters:
            quiet (bool): If True, you will not get printed any message.
        """

        # Writing to file
        with open(self.path_output, "w", encoding="utf-8-sig") as f:
            f.writelines(self.__output + ["\n"])
            if self.__output_extradata:
                f.write("\n[Aegisub Extradata]\n")
                f.writelines(self.__output_extradata)

        self.__saved = True

        if not quiet:
            print(
                "Produced lines: %d\nProcess duration (in seconds): %.3f"
                % (self.__plines, time.time() - self.__ptime)
            )

    def open_aegisub(self) -> int:
        """Open the output (specified in self.path_output) with Aegisub.

        This can be usefull if you don't have MPV installed or you want to look at your output in detailed.

        Returns:
            0 if success, -1 if the output couldn't be opened.
        """

        # Check if it was saved
        if not self.__saved:
            print(
                "[WARNING] You've tried to open the output with Aegisub before having saved. Check your code."
            )
            return -1

        if sys.platform == "win32":
            os.startfile(self.path_output)
        else:
            try:
                subprocess.call(["aegisub", os.path.abspath(self.path_output)])
            except FileNotFoundError:
                print("[WARNING] Aegisub not found.")
                return -1

        return 0

    def open_mpv(
        self, video_path: str = "", video_start: str = "", full_screen: bool = False
    ) -> int:
        """Open the output (specified in self.path_output) in softsub with the MPV player.
        To utilize this function, MPV player is required. Additionally if you're on Windows,
        MPV must be in the PATH (check https://pyonfx.readthedocs.io/en/latest/quick%20start.html#installation-extra-step).

        This is one of the fastest way to reproduce your output in a comfortable way.

        Parameters:
            video_path (string): The video file path (absolute) to reproduce. If not specified, **meta.video** is automatically taken.
            video_start (string): The start time for the video (more info: https://mpv.io/manual/master/#options-start). If not specified, 0 is automatically taken.
            full_screen (bool): If True, it will reproduce the output in full screen. If not specified, False is automatically taken.
        """

        # Check if it was saved
        if not self.__saved:
            print(
                "[ERROR] You've tried to open the output with MPV before having saved. Check your code."
            )
            return -1

        # Check if mpv is usable
        if self.meta.video.startswith("?dummy") and not video_path:
            print(
                "[WARNING] Cannot use MPV (if you have it in your PATH) for file preview, since your .ass contains a dummy video.\n"
                "You can specify a new video source using video_path parameter, check the documentation of the function."
            )
            return -1

        # Setting up the command to execute
        cmd = ["mpv"]

        if not video_path:
            cmd.append(self.meta.video)
        else:
            cmd.append(video_path)
        if video_start:
            cmd.append("--start=" + video_start)
        if full_screen:
            cmd.append("--fs")

        cmd.append("--sub-file=" + self.path_output)

        try:
            subprocess.call(cmd)
        except FileNotFoundError:
            print(
                "[WARNING] MPV not found in your environment variables.\n"
                "Please refer to the documentation's \"Quick Start\" section if you don't know how to solve it."
            )
            return -1

        return 0


def pretty_print(
    obj: Union[Meta, Style, Line, Word, Syllable, Char], indent: int = 0, name: str = ""
) -> str:
    # Utility function to print object Meta, Style, Line, Word, Syllable and Char (this is a dirty solution probably)
    if type(obj) == Line:
        out = " " * indent + f"lines[{obj.i}] ({type(obj).__name__}):\n"
    elif type(obj) == Word:
        out = " " * indent + f"words[{obj.i}] ({type(obj).__name__}):\n"
    elif type(obj) == Syllable:
        out = " " * indent + f"syls[{obj.i}] ({type(obj).__name__}):\n"
    elif type(obj) == Char:
        out = " " * indent + f"chars[{obj.i}] ({type(obj).__name__}):\n"
    else:
        out = " " * indent + f"{name}({type(obj).__name__}):\n"

    # Let's print all this object fields
    indent += 4
    for k, v in obj.__dict__.items():
        if "__dict__" in dir(v):
            # Work recursively to print another object
            out += pretty_print(v, indent, k + " ")
        elif type(v) == list:
            for i, el in enumerate(v):
                # Work recursively to print other objects inside a list
                out += pretty_print(el, indent, f"{k}[{i}] ")
        else:
            # Just print a field of this object
            out += " " * indent + f"{k}: {str(v)}\n"

    return out
