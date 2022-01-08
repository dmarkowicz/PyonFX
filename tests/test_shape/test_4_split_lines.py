# flake8: noqa

import pytest_check as check
from pyonfx import Shape


def test_split_lines1() -> None:
    original = Shape.from_ass_string("m 0 0 l 100 0")
    dest = Shape.from_ass_string(
        "m 0 0 l 14.286 0 28.571 0 42.857 0 57.143 0 71.429 0 85.714 0 100 0"
    )
    original.split_lines()
    original.round()
    check.equal(original, dest)


def test_split_lines2() -> None:
    original = Shape.from_ass_string(
        "m 0 0 l 200 200 l 300 30 m -300 -100 l 200 300 l 300 300 l 500 600"
    )
    dest = Shape.from_ass_string(
        "m 0 0 l 11.111 11.111 22.222 22.222 33.333 33.333 44.444 44.444 55.556 55.556 66.667 66.667 77.778 77.778 88.889 88.889 100 100 111.111 111.111 122.222 122.222 133.333 133.333 144.444 144.444 155.556 155.556 166.667 166.667 177.778 177.778 188.889 188.889 200 200 207.692 186.923 215.385 173.846 223.077 160.769 230.769 147.692 238.462 134.615 246.154 121.538 253.846 108.462 261.538 95.385 269.231 82.308 276.923 69.231 284.615 56.154 292.308 43.077 300 30 m -300 -100 l -287.805 -90.244 -275.61 -80.488 -263.415 -70.732 -251.22 -60.976 -239.024 -51.22 -226.829 -41.463 -214.634 -31.707 -202.439 -21.951 -190.244 -12.195 -178.049 -2.439 -165.854 7.317 -153.659 17.073 -141.463 26.829 -129.268 36.585 -117.073 46.341 -104.878 56.098 -92.683 65.854 -80.488 75.61 -68.293 85.366 -56.098 95.122 -43.902 104.878 -31.707 114.634 -19.512 124.39 -7.317 134.146 4.878 143.902 17.073 153.659 29.268 163.415 41.463 173.171 53.659 182.927 65.854 192.683 78.049 202.439 90.244 212.195 102.439 221.951 114.634 231.707 126.829 241.463 139.024 251.22 151.22 260.976 163.415 270.732 175.61 280.488 187.805 290.244 200 300 214.286 300 228.571 300 242.857 300 257.143 300 271.429 300 285.714 300 300 300 308.696 313.043 317.391 326.087 326.087 339.13 334.783 352.174 343.478 365.217 352.174 378.261 360.87 391.304 369.565 404.348 378.261 417.391 386.957 430.435 395.652 443.478 404.348 456.522 413.043 469.565 421.739 482.609 430.435 495.652 439.13 508.696 447.826 521.739 456.522 534.783 465.217 547.826 473.913 560.87 482.609 573.913 491.304 586.957 500 600"
    )
    original.split_lines()
    original.round()
    check.equal(original, dest)


def test_split_lines3() -> None:
    original = Shape.from_ass_string("m -100.5 0 l 100 0 b 100 100 -100 100 -100.5 0 c")
    dest = Shape.from_ass_string(
        "m -100.5 0 l -85.077 0 -69.654 0 -54.231 0 -38.808 0 -23.385 0 -7.962 0 7.462 0 22.885 0 38.308 0 53.731 0 69.154 0 84.577 0 100 0 99.964 2.325 99.855 4.614 99.676 6.866 99.426 9.082 99.108 11.261 98.723 13.403 98.271 15.509 97.754 17.578 97.173 19.611 96.528 21.606 95.822 23.566 95.056 25.488 94.23 27.374 93.345 29.224 92.403 31.036 91.405 32.812 90.352 34.552 89.246 36.255 88.086 37.921 86.876 39.551 85.614 41.144 84.304 42.7 82.945 44.22 81.54 45.703 80.088 47.15 78.592 48.56 77.053 49.933 75.471 51.27 73.848 52.57 72.184 53.833 70.482 55.06 68.742 56.25 66.965 57.404 65.153 58.521 63.307 59.601 61.427 60.645 59.515 61.652 57.572 62.622 55.599 63.556 53.598 64.453 51.569 65.314 49.514 66.138 47.433 66.925 45.329 67.676 43.201 68.39 41.052 69.067 38.882 69.708 36.692 70.312 34.484 70.88 32.259 71.411 27.762 72.363 23.209 73.169 18.61 73.828 13.975 74.341 9.311 74.707 4.629 74.927 -0.062 75 -4.755 74.927 -9.438 74.707 -14.103 74.341 -18.741 73.828 -23.343 73.169 -27.9 72.363 -32.402 71.411 -34.63 70.88 -36.841 70.312 -39.033 69.708 -41.207 69.067 -43.359 68.39 -45.49 67.676 -47.599 66.925 -49.683 66.138 -51.743 65.314 -53.776 64.453 -55.782 63.556 -57.759 62.622 -59.707 61.652 -61.624 60.645 -63.509 59.601 -65.361 58.521 -67.178 57.404 -68.961 56.25 -70.707 55.06 -72.415 53.833 -74.085 52.57 -75.714 51.27 -77.303 49.933 -78.85 48.56 -80.353 47.15 -81.811 45.703 -83.224 44.22 -84.59 42.7 -85.909 41.144 -87.178 39.551 -88.397 37.921 -89.564 36.255 -90.68 34.552 -91.741 32.812 -92.748 31.036 -93.699 29.224 -94.593 27.374 -95.428 25.488 -96.205 23.566 -96.92 21.606 -97.575 19.611 -98.166 17.578 -98.693 15.509 -99.156 13.403 -99.552 11.261 -99.881 9.082 -100.141 6.866 -100.332 4.614 -100.452 2.325 -100.5 0"
    )
    original.split_lines()
    original.round()
    check.equal(original, dest)