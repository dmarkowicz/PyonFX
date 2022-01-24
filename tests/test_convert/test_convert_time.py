
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest_check as check
import pytest
from pyonfx import Ass, ConvertTime

folder = Path(__file__).parent


@pytest.mark.skip
class _Process:
    __slots__ = 'p'
    p: str | Path

    def __init__(self, p: str | Path) -> None:
        self.p = p

    @property
    def originals(self) -> List[str]:
        return [
            ltext
            for s in self.io._sections.values()
            if s.name == '[Events]'
            for ltext in s.text.strip().splitlines()[1:]
        ]

    @property
    def io(self) -> Ass:
        return Ass(self.p, 'out_test.ass', extended=False)


def test_convert_time0() -> None:
    p = _Process(folder / 'originals_lines.ass')
    originals = p.originals
    transformeds = [line.compose_ass_line() for line in p.io.lines]

    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):
        check.equal(
            ConvertTime.ts2seconds(origin.split(',')[1]),
            ConvertTime.ts2seconds(transformed.split(',')[1]),
            msg='Start line n°' + str(i)
        )
        check.equal(
            ConvertTime.ts2seconds(origin.split(',')[2]),
            ConvertTime.ts2seconds(transformed.split(',')[2]),
            msg='End line n°' + str(i)
        )


def test_convert_time1() -> None:
    originals = _Process(folder / 'originals_lines.ass').originals
    transformeds = [
        line.compose_ass_line()
        for line in _Process(folder / 'originals_lines+0.01.ass').io.lines
    ]

    # Exclude the first line since it's fucked
    for i, (origin, transformed) in enumerate(zip(originals[1:], transformeds[1:]), start=1):
        check.equal(
            ConvertTime.ts2seconds(origin.split(',')[1]),
            ConvertTime.ts2seconds(transformed.split(',')[1]),
            msg='Start line n°' + str(i)
        )
        check.equal(
            ConvertTime.ts2seconds(origin.split(',')[2]),
            ConvertTime.ts2seconds(transformed.split(',')[2]),
            msg='End line n°' + str(i)
        )


def test_convert_time2() -> None:
    originals = _Process(folder / 'originals_lines.ass').originals
    transformeds = [
        line.compose_ass_line()
        for line in _Process(folder / 'originals_lines+0.02.ass').io.lines
    ]

    # Exclude the first line since it's fucked
    for i, (origin, transformed) in enumerate(zip(originals[1:], transformeds[1:]), start=1):
        check.equal(
            ConvertTime.ts2seconds(origin.split(',')[1]),
            ConvertTime.ts2seconds(transformed.split(',')[1]),
            msg='Start line n°' + str(i)
        )
        check.equal(
            ConvertTime.ts2seconds(origin.split(',')[2]),
            ConvertTime.ts2seconds(transformed.split(',')[2]),
            msg='End line n°' + str(i)
        )


def test_convert_time3() -> None:
    originals = _Process(folder / 'originals_lines+1frame.ass').originals
    transformeds = [
        line.compose_ass_line()
        for line in _Process(folder / 'originals_lines+0.03.ass').io.lines
    ]

    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):
        if i not in range(206, 5000, 240):
            check.equal(
                ConvertTime.ts2seconds(origin.split(',')[1]),
                ConvertTime.ts2seconds(transformed.split(',')[1]),
                msg='Start line n°' + str(i)
            )
        if i not in range(205, 5000, 240):
            check.equal(
                ConvertTime.ts2seconds(origin.split(',')[2]),
                ConvertTime.ts2seconds(transformed.split(',')[2]),
                msg='End line n°' + str(i)
            )


def test_convert_time4() -> None:
    originals = _Process(folder / 'originals_lines+1frame.ass').originals
    transformeds = [
        line.compose_ass_line()
        for line in _Process(folder / 'originals_lines+0.04.ass').io.lines
    ]
    # p = _Process(folder / 'originals_lines+0.04.ass')
    # p.io.save(p.io.lines, False)
    # p.io.open_aegisub()
    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):
        check.equal(
            ConvertTime.ts2seconds(origin.split(',')[1]),
            ConvertTime.ts2seconds(transformed.split(',')[1]),
            msg='Start line n°' + str(i)
        )
        check.equal(
            ConvertTime.ts2seconds(origin.split(',')[2]),
            ConvertTime.ts2seconds(transformed.split(',')[2]),
            msg='End line n°' + str(i)
        )
