
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import pytest_check as check
from pyonfx import Ass, ConvertTime, logger

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
    transformeds = [line.as_text() for line in p.io.lines]

    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):

        _, ostart, oend, *_ = origin.split(',')
        _, tstart, tend, *_ = transformed.split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)

        check.equal(
            ConvertTime.ts2seconds(ostart),
            ConvertTime.ts2seconds(tstart),
            msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
        )
        check.equal(
            ConvertTime.ts2seconds(oend),
            ConvertTime.ts2seconds(tend),
            msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
        )


def test_convert_time1() -> None:
    originals = _Process(folder / 'originals_lines.ass').originals
    transformeds = [
        line.as_text()
        for line in _Process(folder / 'originals_lines+0.01.ass').io.lines
    ]

    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):
        # Exclude the first line since it's fucked
        if i == 1:
            continue

        _, ostart, oend, *_ = origin.split(',')
        _, tstart, tend, *_ = transformed.split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)

        check.almost_equal(
            ConvertTime.ts2seconds(ostart),
            ConvertTime.ts2seconds(tstart),
            abs=0.01001,
            msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
        )
        check.almost_equal(
            ConvertTime.ts2seconds(oend),
            ConvertTime.ts2seconds(tend),
            abs=0.01001,
            msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
        )


def test_convert_time2() -> None:
    originals = _Process(folder / 'originals_lines.ass').originals
    transformeds = [
        line.as_text()
        for line in _Process(folder / 'originals_lines+0.02.ass').io.lines
    ]

    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):
        # Exclude the first line since it's fucked
        if i == 1:
            continue

        _, ostart, oend, *_ = origin.split(',')
        _, tstart, tend, *_ = transformed.split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)

        check.almost_equal(
            ConvertTime.ts2seconds(ostart),
            ConvertTime.ts2seconds(tstart),
            abs=0.01001,
            msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
        )
        check.almost_equal(
            ConvertTime.ts2seconds(oend),
            ConvertTime.ts2seconds(tend),
            abs=0.01001,
            msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
        )



def test_convert_time3() -> None:
    originals = _Process(folder / 'originals_lines+1frame.ass').originals
    transformeds = [
        line.as_text()
        for line in _Process(folder / 'originals_lines+0.03.ass').io.lines
    ]

    broken_endl = list(range(164, 5000, 240)) + list(range(205, 5000, 240))
    broken_startl = [x + 1 for x in broken_endl]

    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):
        # Exclude the first line since it's fucked
        if i == 1:
            continue

        _, ostart, oend, *_ = origin.split(',')
        _, tstart, tend, *_ = transformed.split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)


        if i not in broken_startl:
            check.almost_equal(
                ConvertTime.ts2seconds(ostart),
                ConvertTime.ts2seconds(tstart),
                abs=0.01001,
                msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
            )
        if i not in broken_endl:
            check.almost_equal(
                ConvertTime.ts2seconds(oend),
                ConvertTime.ts2seconds(tend),
                abs=0.01001,
                msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
            )


def test_convert_time4() -> None:
    originals = _Process(folder / 'originals_lines+1frame.ass').originals
    transformeds = [
        line.as_text()
        for line in _Process(folder / 'originals_lines+0.04.ass').io.lines
    ]
    # p = _Process(folder / 'originals_lines+0.04.ass')
    # p.io.save(p.io.lines, False)
    # p.io.open_aegisub()
    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):

        _, ostart, oend, *_ = origin.split(',')
        _, tstart, tend, *_ = transformed.split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)

        check.almost_equal(
            ConvertTime.ts2seconds(ostart),
            ConvertTime.ts2seconds(tstart),
            abs=0.01001,
            msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
        )
        check.almost_equal(
            ConvertTime.ts2seconds(oend),
            ConvertTime.ts2seconds(tend),
            abs=0.01001,
            msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
        )


def test_convert_time5() -> None:
    originals = _Process(folder / 'moi_test_expected.ass').originals
    transformeds = [
        line.as_text()
        for line in _Process(folder / 'moi_test.ass').io.lines
    ]
    for i, (origin, transformed) in enumerate(zip(originals, transformeds), start=1):

        _, ostart, oend, *_ = origin.split(',')
        _, tstart, tend, *_ = transformed.split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)

        check.almost_equal(
            ConvertTime.ts2seconds(ostart),
            ConvertTime.ts2seconds(tstart),
            abs=0.01001,
            msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
        )
        check.almost_equal(
            ConvertTime.ts2seconds(oend),
            ConvertTime.ts2seconds(tend),
            abs=0.01001,
            msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
        )


if __name__ == '__main__':
    logger.set_level(10)
    test_convert_time0()
    test_convert_time1()
    test_convert_time2()
    test_convert_time3()
    test_convert_time4()
    test_convert_time5()
