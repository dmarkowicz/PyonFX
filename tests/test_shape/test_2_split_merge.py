# flake8: noqa
import pytest_check as check
from pyonfx import Shape


def test_split_shape1() -> None:
    original = Shape.from_ass_string('m 14.266 35.891 b 14.047 35.891 13.859 35.953 13.734 36.109 13.609 36.25 13.547 36.422 13.547 36.609 l 13.547 52.141 b 13.547 52.578 13.422 52.906 13.188 53.109 12.953 53.312 12.562 53.406 12.047 53.406 l 5.094 53.406 b 4.641 53.406 4.328 53.312 4.141 53.125 3.969 52.953 3.875 52.656 3.875 52.25 l 3.875 13.438 b 3.875 12.734 4.172 12.391 4.75 12.391 l 12.656 12.391 b 13.25 12.391 13.547 12.719 13.547 13.391 l 13.547 27.531 b 13.547 28.156 13.828 28.469 14.375 28.469 l 27.141 28.469 b 27.734 28.469 28.031 28.188 28.031 27.594 l 28.031 13.391 b 28.031 12.719 28.297 12.391 28.859 12.391 l 36.766 12.391 b 37.391 12.391 37.703 12.734 37.703 13.438 l 37.703 52.25 b 37.703 53.016 37.297 53.406 36.484 53.406 l 29.469 53.406 b 28.953 53.406 28.578 53.312 28.359 53.125 28.141 52.953 28.031 52.609 28.031 52.141 l 28.031 36.609 b  28.031 36.375 27.969 36.203 27.828 36.078 27.703 35.953 27.516 35.891 27.25 35.891 l 14.266 35.891 m 58.047 53.953 b 55.141 53.953 52.562 53.328 50.328 52.047 48.109 50.781 46.359 49 45.109 46.719 43.859 44.438 43.234 41.828 43.234 38.922 43.234 35.906 43.859 33.234 45.141 30.906 46.406 28.594 48.172 26.781 50.438 25.484 52.703 24.203 55.266 23.562 58.094 23.562 61.016 23.562 63.594 24.203 65.844 25.516 68.094 26.828 69.828 28.641 71.062 30.969 72.297 33.281 72.922 35.922 72.922 38.875 72.922 41.781 72.297 44.375 71.031 46.656 69.781 48.953 68.031 50.734 65.781 52.031 63.531 53.312 60.953 53.953 58.047 53.953 l 58.047 53.953 m 58.094 48.328 b 60.125 48.328 61.672 47.516 62.719 45.922 63.766 44.312 64.297 41.922 64.297 38.75 64.297 35.547 63.766 33.125 62.719 31.484 61.672 29.844 60.125 29.031 58.094 29.031 56.078 29.031 54.531 29.844 53.453 31.484 52.391 33.125 51.859 35.547 51.859 38.75 51.859 41.891 52.391 44.266 53.453 45.891 54.531 47.516 56.078 48.328 58.094 48.328 l 58.094 48.328 m 80.375 62.531 b 79.156 62.531 78.312 62.469 77.828 62.328 77.359 62.203 77.109 61.922 77.109 61.484 l 77.109 57.219 b 77.109 56.625 77.641 56.344 78.719 56.344 l 82.469 56.344 b 83.578 56.344 84.547 55.922 85.344 55.094 86.156 54.266 86.562 53.391 86.562 52.469 86.562 52.141 86.516 51.844 86.406 51.578 l 75.781 25.438 b 75.719 25.219 75.672 25.062 75.672 25 75.672 24.734 75.766 24.531 75.953 24.359 76.141 24.188 76.375 24.109 76.672 24.109 l 83.641 24.109 b 83.891 24.109 84.141 24.188 84.375 24.359 84.625 24.531 84.781 24.75 84.859 25.047 l 90.984 41.516 b 91.094 41.891 91.25 42.078 91.438 42.078 91.578 42.078 91.75 41.875 91.922 41.469 l 98.016 25  b 98.266 24.406 98.672 24.109 99.219 24.109 l 103.984 24.109 b 104.234 24.109 104.453 24.188 104.609 24.359 104.781 24.531 104.859 24.719 104.859 24.938 104.859 25.016 104.828 25.156 104.75 25.375 l 93.422 53.406 b 92.391 55.953 91.297 57.859 90.156 59.156 89.016 60.453 87.703 61.328 86.203 61.812 84.719 62.297 82.766 62.531 80.375 62.531 l 80.375 62.531 m 124.719 50.594 b 122.281 52.844 119.375 53.953 115.984 53.953 113.219 53.953 111.031 53.219 109.438 51.719 107.828 50.234 107.031 48.281 107.031 45.891 107.031 42.75 108.438 40.266 111.25 38.406 114.078 36.531 117.984 35.547 123 35.438 l 124.938 35.281 b 125.531 35.281 125.828 35.016 125.828 34.5 l 125.828 33.344 b 125.828 31.906 125.422 30.797 124.609 30.031 123.797 29.25 122.641 28.859 121.125 28.859 118.547 28.859 116.625 29.969 115.375 32.172 115.234 32.547 114.984 32.734 114.656 32.734 114.469 32.734 114.312 32.719 114.156 32.672 l 108.625 31.406 b 108.297 31.328 108.141 31.125 108.141 30.797 108.141 30.578 108.203 30.297 108.359 29.969 109.234 27.828 110.812 26.234 113.047 25.156 115.297 24.094 118.125 23.562 121.516 23.562 129.797 23.562 133.953 26.641 133.953 32.781 l 133.953 52.188 b 133.953 52.641 133.875 52.953 133.734 53.125 133.578 53.312 133.312 53.406 132.906 53.406 l 127.203 53.406 b 126.844 53.406 126.562 53.281 126.406 53.016 126.234 52.766 126.125 52.375 126.047 51.859 l 125.938 50.641 b 125.891 50.312 125.766 50.141 125.547 50.141 125.406 50.141 125.125 50.297 124.719 50.594 l 124.719 50.594 m 125.828 40.359 b 125.828 40.031 125.766 39.797 125.625 39.641 125.5 39.5 125.281 39.422 125 39.422 l 123.453 39.531 b 121.312 39.641 119.422 40.156 117.781 41.047 116.141 41.953 115.312 43.203 115.312 44.781 115.312 45.891 115.703 46.766 116.484 47.406 117.25 48.062 118.328 48.375 119.688 48.375 121.312 48.375 122.75 47.875 124 46.891 125.219 45.859 125.828 44.875 125.828 43.953 l 125.828 40.359 125.828 40.359 m 141.297 53.406 b 140.906 53.406 140.609 53.328 140.422 53.156 140.234 53 140.141 52.703 140.141 52.297 l 140.141 13.438 b 140.141 12.734 140.438 12.391 141.031 12.391 l  147.656 12.391 b 148.219 12.391 148.484 12.703 148.484 13.328 l 148.484 27.484 b 148.484 27.781 148.578 27.922 148.766 27.922 148.875 27.922 149 27.844 149.156 27.703 150.844 26.188 152.422 25.125 153.875 24.5 155.344 23.875 156.875 23.562 158.5 23.562 161.297 23.562 163.453 24.328 164.969 25.875 166.469 27.422 167.234 29.594 167.234 32.406 l 167.234 52.078 b 167.234 52.969 166.781 53.406 165.906 53.406 l 159.766 53.406 b 159.359 53.406 159.094 53.328 158.938 53.156 158.797 53 158.719 52.703 158.719 52.297 l 158.719 33.609 b 158.719 32.359 158.375 31.391 157.688 30.719 157.016 30.031 156.031 29.688 154.734 29.688 153.594 29.688 152.562 29.969 151.641 30.516 150.719 31.078 149.719 31.938 148.656 33.125 l 148.656 52.141 b 148.656 52.578 148.547 52.906 148.328 53.109 148.109 53.312 147.734 53.406 147.219 53.406 l 141.297  53.406 141.297 53.406 m 186.578 53.953 b 183.672 53.953 181.094 53.328 178.859 52.047 176.641 50.781 174.891 49 173.641 46.719 172.391 44.438 171.766 41.828 171.766 38.922 171.766 35.906 172.391 33.234 173.672 30.906 174.938 28.594 176.703 26.781 178.969 25.484 181.234 24.203 183.797 23.562 186.625 23.562 189.547 23.562 192.125 24.203 194.375 25.516 196.625 26.828 198.359 28.641 199.594 30.969 200.828 33.281 201.453 35.922 201.453 38.875 201.453 41.781 200.828 44.375 199.562 46.656 198.312 48.953 196.562 50.734 194.312 52.031 192.062 53.312 189.484 53.953 186.578 53.953 l 186.578 53.953 m 186.625 48.328 b 188.656 48.328 190.203 47.516 191.25 45.922 192.297 44.312 192.828 41.922 192.828 38.75 192.828 35.547 192.297 33.125 191.25 31.484 190.203 29.844 188.656 29.031 186.625 29.031 184.609 29.031 183.062 29.844 181.984 31.484 180.922 33.125 180.391 35.547 180.391 38.75 180.391 41.891 180.922 44.266 181.984 45.891 183.062 47.516 184.609 48.328 186.625 48.328 l 186.625 48.328')
    dests = [
        Shape.from_ass_string('m 14.266 35.891 b 14.047 35.891 13.859 35.953 13.734 36.109 13.609 36.25 13.547 36.422 13.547 36.609 l 13.547 52.141 b 13.547 52.578 13.422 52.906 13.188 53.109 12.953 53.312 12.562 53.406 12.047 53.406 l 5.094 53.406 b 4.641 53.406 4.328 53.312 4.141 53.125 3.969 52.953 3.875 52.656 3.875 52.25 l 3.875 13.438 b 3.875 12.734 4.172 12.391 4.75 12.391 l 12.656 12.391 b 13.25 12.391 13.547 12.719 13.547 13.391 l 13.547 27.531 b 13.547 28.156 13.828 28.469 14.375 28.469 l 27.141 28.469 b 27.734 28.469 28.031 28.188 28.031 27.594 l 28.031 13.391 b 28.031 12.719 28.297 12.391 28.859 12.391 l 36.766 12.391 b 37.391 12.391 37.703 12.734 37.703 13.438 l 37.703 52.25 b 37.703 53.016 37.297 53.406 36.484 53.406 l 29.469 53.406 b 28.953 53.406 28.578 53.312 28.359 53.125 28.141 52.953 28.031 52.609 28.031 52.141 l 28.031 36.609 b 28.031 36.375 27.969 36.203 27.828 36.078 27.703 35.953 27.516 35.891 27.25 35.891 l 14.266 35.891'),
        Shape.from_ass_string('m 58.047 53.953 b 55.141 53.953 52.562 53.328 50.328 52.047 48.109 50.781 46.359 49 45.109 46.719 43.859 44.438 43.234 41.828 43.234 38.922 43.234 35.906 43.859 33.234 45.141 30.906 46.406 28.594 48.172 26.781 50.438 25.484 52.703 24.203 55.266 23.562 58.094 23.562 61.016 23.562 63.594 24.203 65.844 25.516 68.094 26.828 69.828 28.641 71.062 30.969 72.297 33.281 72.922 35.922 72.922 38.875 72.922 41.781 72.297 44.375 71.031 46.656 69.781 48.953 68.031 50.734 65.781 52.031 63.531 53.312 60.953 53.953 58.047 53.953 l 58.047 53.953'),
        Shape.from_ass_string('m 58.094 48.328 b 60.125 48.328 61.672 47.516 62.719 45.922 63.766 44.312 64.297 41.922 64.297 38.75 64.297 35.547 63.766 33.125 62.719 31.484 61.672 29.844 60.125 29.031 58.094 29.031 56.078 29.031 54.531 29.844 53.453 31.484 52.391 33.125 51.859 35.547 51.859 38.75 51.859 41.891 52.391 44.266 53.453 45.891 54.531 47.516 56.078 48.328 58.094 48.328 l 58.094 48.328'),
        Shape.from_ass_string('m 80.375 62.531 b 79.156 62.531 78.312 62.469 77.828 62.328 77.359 62.203 77.109 61.922 77.109 61.484 l 77.109 57.219 b 77.109 56.625 77.641 56.344 78.719 56.344 l 82.469 56.344 b 83.578 56.344 84.547 55.922 85.344 55.094 86.156 54.266 86.562 53.391 86.562 52.469 86.562 52.141 86.516 51.844 86.406 51.578 l 75.781 25.438 b 75.719 25.219 75.672 25.062 75.672 25 75.672 24.734 75.766 24.531 75.953 24.359 76.141 24.188 76.375 24.109 76.672 24.109 l 83.641 24.109 b 83.891 24.109 84.141 24.188 84.375 24.359 84.625 24.531 84.781 24.75 84.859 25.047 l 90.984 41.516 b 91.094 41.891 91.25 42.078 91.438 42.078 91.578 42.078 91.75 41.875 91.922 41.469 l 98.016 25 b 98.266 24.406 98.672 24.109 99.219 24.109 l 103.984 24.109 b 104.234 24.109 104.453 24.188 104.609 24.359 104.781 24.531 104.859 24.719 104.859 24.938 104.859 25.016 104.828 25.156 104.75 25.375 l 93.422 53.406 b 92.391 55.953 91.297 57.859 90.156 59.156 89.016 60.453 87.703 61.328 86.203 61.812 84.719 62.297 82.766 62.531 80.375 62.531 l 80.375 62.531'),
        Shape.from_ass_string('m 124.719 50.594 b 122.281 52.844 119.375 53.953 115.984 53.953 113.219 53.953 111.031 53.219 109.438 51.719 107.828 50.234 107.031 48.281 107.031 45.891 107.031 42.75 108.438 40.266 111.25 38.406 114.078 36.531 117.984 35.547 123 35.438 l 124.938 35.281 b 125.531 35.281 125.828 35.016 125.828 34.5 l 125.828 33.344 b 125.828 31.906 125.422 30.797 124.609 30.031 123.797 29.25 122.641 28.859 121.125 28.859 118.547 28.859 116.625 29.969 115.375 32.172 115.234 32.547 114.984 32.734 114.656 32.734 114.469 32.734 114.312 32.719 114.156 32.672 l 108.625 31.406 b 108.297 31.328 108.141 31.125 108.141 30.797 108.141 30.578 108.203 30.297 108.359 29.969 109.234 27.828 110.812 26.234 113.047 25.156 115.297 24.094 118.125 23.562 121.516 23.562 129.797 23.562 133.953 26.641 133.953 32.781 l 133.953 52.188 b 133.953 52.641 133.875 52.953 133.734 53.125 133.578 53.312 133.312 53.406 132.906 53.406 l 127.203 53.406 b 126.844 53.406 126.562 53.281 126.406 53.016 126.234 52.766 126.125 52.375 126.047 51.859 l 125.938 50.641 b 125.891 50.312 125.766 50.141 125.547 50.141 125.406 50.141 125.125 50.297 124.719 50.594 l 124.719 50.594'),
        Shape.from_ass_string('m 125.828 40.359 b 125.828 40.031 125.766 39.797 125.625 39.641 125.5 39.5 125.281 39.422 125 39.422 l 123.453 39.531 b 121.312 39.641 119.422 40.156 117.781 41.047 116.141 41.953 115.312 43.203 115.312 44.781 115.312 45.891 115.703 46.766 116.484 47.406 117.25 48.062 118.328 48.375 119.688 48.375 121.312 48.375 122.75 47.875 124 46.891 125.219 45.859 125.828 44.875 125.828 43.953 l 125.828 40.359 125.828 40.359'),
        Shape.from_ass_string('m 141.297 53.406 b 140.906 53.406 140.609 53.328 140.422 53.156 140.234 53 140.141 52.703 140.141 52.297 l 140.141 13.438 b 140.141 12.734 140.438 12.391 141.031 12.391 l 147.656 12.391 b 148.219 12.391 148.484 12.703 148.484 13.328 l 148.484 27.484 b 148.484 27.781 148.578 27.922 148.766 27.922 148.875 27.922 149 27.844 149.156 27.703 150.844 26.188 152.422 25.125 153.875 24.5 155.344 23.875 156.875 23.562 158.5 23.562 161.297 23.562 163.453 24.328 164.969 25.875 166.469 27.422 167.234 29.594 167.234 32.406 l 167.234 52.078 b 167.234 52.969 166.781 53.406 165.906 53.406 l 159.766 53.406 b 159.359 53.406 159.094 53.328 158.938 53.156 158.797 53 158.719 52.703 158.719 52.297 l 158.719 33.609 b 158.719 32.359 158.375 31.391 157.688 30.719 157.016 30.031 156.031 29.688 154.734 29.688 153.594 29.688 152.562 29.969 151.641 30.516 150.719 31.078 149.719 31.938 148.656 33.125 l 148.656 52.141 b 148.656 52.578 148.547 52.906 148.328 53.109 148.109 53.312 147.734 53.406 147.219 53.406 l 141.297 53.406 141.297 53.406'),
        Shape.from_ass_string('m 186.578 53.953 b 183.672 53.953 181.094 53.328 178.859 52.047 176.641 50.781 174.891 49 173.641 46.719 172.391 44.438 171.766 41.828 171.766 38.922 171.766 35.906 172.391 33.234 173.672 30.906 174.938 28.594 176.703 26.781 178.969 25.484 181.234 24.203 183.797 23.562 186.625 23.562 189.547 23.562 192.125 24.203 194.375 25.516 196.625 26.828 198.359 28.641 199.594 30.969 200.828 33.281 201.453 35.922 201.453 38.875 201.453 41.781 200.828 44.375 199.562 46.656 198.312 48.953 196.562 50.734 194.312 52.031 192.062 53.312 189.484 53.953 186.578 53.953 l 186.578 53.953'),
        Shape.from_ass_string('m 186.625 48.328 b 188.656 48.328 190.203 47.516 191.25 45.922 192.297 44.312 192.828 41.922 192.828 38.75 192.828 35.547 192.297 33.125 191.25 31.484 190.203 29.844 188.656 29.031 186.625 29.031 184.609 29.031 183.062 29.844 181.984 31.484 180.922 33.125 180.391 35.547 180.391 38.75 180.391 41.891 180.922 44.266 181.984 45.891 183.062 47.516 184.609 48.328 186.625 48.328 l 186.625 48.328')
    ]
    for dest in dests:
        dest.round()
    shapes = original.split_shape()
    check.equal(shapes, dests)


def test_split_shape2() -> None:
    original = Shape.from_ass_string('m 930 676 l 1078 824 1228 782 1078 612 788 668 632 860')
    dest = Shape.from_ass_string('m 930 676 l 1078 824 1228 782 1078 612 788 668 632 860')
    shape_s = original.split_shape()
    check.equal(len(shape_s), 1)
    check.equal(shape_s.pop(0), dest)


def test_merge_shapes() -> None:
    originals = [
        Shape.from_ass_string('m 14.266 35.891 b 14.047 35.891 13.859 35.953 13.734 36.109 13.609 36.25 13.547 36.422 13.547 36.609 l 13.547 52.141 b 13.547 52.578 13.422 52.906 13.188 53.109 12.953 53.312 12.562 53.406 12.047 53.406 l 5.094 53.406 b 4.641 53.406 4.328 53.312 4.141 53.125 3.969 52.953 3.875 52.656 3.875 52.25 l 3.875 13.438 b 3.875 12.734 4.172 12.391 4.75 12.391 l 12.656 12.391 b 13.25 12.391 13.547 12.719 13.547 13.391 l 13.547 27.531 b 13.547 28.156 13.828 28.469 14.375 28.469 l 27.141 28.469 b 27.734 28.469 28.031 28.188 28.031 27.594 l 28.031 13.391 b 28.031 12.719 28.297 12.391 28.859 12.391 l 36.766 12.391 b 37.391 12.391 37.703 12.734 37.703 13.438 l 37.703 52.25 b 37.703 53.016 37.297 53.406 36.484 53.406 l 29.469 53.406 b 28.953 53.406 28.578 53.312 28.359 53.125 28.141 52.953 28.031 52.609 28.031 52.141 l 28.031 36.609 b 28.031 36.375 27.969 36.203 27.828 36.078 27.703 35.953 27.516 35.891 27.25 35.891 l 14.266 35.891'),
        Shape.from_ass_string('m 58.047 53.953 b 55.141 53.953 52.562 53.328 50.328 52.047 48.109 50.781 46.359 49 45.109 46.719 43.859 44.438 43.234 41.828 43.234 38.922 43.234 35.906 43.859 33.234 45.141 30.906 46.406 28.594 48.172 26.781 50.438 25.484 52.703 24.203 55.266 23.562 58.094 23.562 61.016 23.562 63.594 24.203 65.844 25.516 68.094 26.828 69.828 28.641 71.062 30.969 72.297 33.281 72.922 35.922 72.922 38.875 72.922 41.781 72.297 44.375 71.031 46.656 69.781 48.953 68.031 50.734 65.781 52.031 63.531 53.312 60.953 53.953 58.047 53.953 l 58.047 53.953'),
        Shape.from_ass_string('m 58.094 48.328 b 60.125 48.328 61.672 47.516 62.719 45.922 63.766 44.312 64.297 41.922 64.297 38.75 64.297 35.547 63.766 33.125 62.719 31.484 61.672 29.844 60.125 29.031 58.094 29.031 56.078 29.031 54.531 29.844 53.453 31.484 52.391 33.125 51.859 35.547 51.859 38.75 51.859 41.891 52.391 44.266 53.453 45.891 54.531 47.516 56.078 48.328 58.094 48.328 l 58.094 48.328'),
        Shape.from_ass_string('m 80.375 62.531 b 79.156 62.531 78.312 62.469 77.828 62.328 77.359 62.203 77.109 61.922 77.109 61.484 l 77.109 57.219 b 77.109 56.625 77.641 56.344 78.719 56.344 l 82.469 56.344 b 83.578 56.344 84.547 55.922 85.344 55.094 86.156 54.266 86.562 53.391 86.562 52.469 86.562 52.141 86.516 51.844 86.406 51.578 l 75.781 25.438 b 75.719 25.219 75.672 25.062 75.672 25 75.672 24.734 75.766 24.531 75.953 24.359 76.141 24.188 76.375 24.109 76.672 24.109 l 83.641 24.109 b 83.891 24.109 84.141 24.188 84.375 24.359 84.625 24.531 84.781 24.75 84.859 25.047 l 90.984 41.516 b 91.094 41.891 91.25 42.078 91.438 42.078 91.578 42.078 91.75 41.875 91.922 41.469 l 98.016 25 b 98.266 24.406 98.672 24.109 99.219 24.109 l 103.984 24.109 b 104.234 24.109 104.453 24.188 104.609 24.359 104.781 24.531 104.859 24.719 104.859 24.938 104.859 25.016 104.828 25.156 104.75 25.375 l 93.422 53.406 b 92.391 55.953 91.297 57.859 90.156 59.156 89.016 60.453 87.703 61.328 86.203 61.812 84.719 62.297 82.766 62.531 80.375 62.531 l 80.375 62.531'),
        Shape.from_ass_string('m 124.719 50.594 b 122.281 52.844 119.375 53.953 115.984 53.953 113.219 53.953 111.031 53.219 109.438 51.719 107.828 50.234 107.031 48.281 107.031 45.891 107.031 42.75 108.438 40.266 111.25 38.406 114.078 36.531 117.984 35.547 123 35.438 l 124.938 35.281 b 125.531 35.281 125.828 35.016 125.828 34.5 l 125.828 33.344 b 125.828 31.906 125.422 30.797 124.609 30.031 123.797 29.25 122.641 28.859 121.125 28.859 118.547 28.859 116.625 29.969 115.375 32.172 115.234 32.547 114.984 32.734 114.656 32.734 114.469 32.734 114.312 32.719 114.156 32.672 l 108.625 31.406 b 108.297 31.328 108.141 31.125 108.141 30.797 108.141 30.578 108.203 30.297 108.359 29.969 109.234 27.828 110.812 26.234 113.047 25.156 115.297 24.094 118.125 23.562 121.516 23.562 129.797 23.562 133.953 26.641 133.953 32.781 l 133.953 52.188 b 133.953 52.641 133.875 52.953 133.734 53.125 133.578 53.312 133.312 53.406 132.906 53.406 l 127.203 53.406 b 126.844 53.406 126.562 53.281 126.406 53.016 126.234 52.766 126.125 52.375 126.047 51.859 l 125.938 50.641 b 125.891 50.312 125.766 50.141 125.547 50.141 125.406 50.141 125.125 50.297 124.719 50.594 l 124.719 50.594'),
        Shape.from_ass_string('m 125.828 40.359 b 125.828 40.031 125.766 39.797 125.625 39.641 125.5 39.5 125.281 39.422 125 39.422 l 123.453 39.531 b 121.312 39.641 119.422 40.156 117.781 41.047 116.141 41.953 115.312 43.203 115.312 44.781 115.312 45.891 115.703 46.766 116.484 47.406 117.25 48.062 118.328 48.375 119.688 48.375 121.312 48.375 122.75 47.875 124 46.891 125.219 45.859 125.828 44.875 125.828 43.953 l 125.828 40.359 125.828 40.359'),
        Shape.from_ass_string('m 141.297 53.406 b 140.906 53.406 140.609 53.328 140.422 53.156 140.234 53 140.141 52.703 140.141 52.297 l 140.141 13.438 b 140.141 12.734 140.438 12.391 141.031 12.391 l 147.656 12.391 b 148.219 12.391 148.484 12.703 148.484 13.328 l 148.484 27.484 b 148.484 27.781 148.578 27.922 148.766 27.922 148.875 27.922 149 27.844 149.156 27.703 150.844 26.188 152.422 25.125 153.875 24.5 155.344 23.875 156.875 23.562 158.5 23.562 161.297 23.562 163.453 24.328 164.969 25.875 166.469 27.422 167.234 29.594 167.234 32.406 l 167.234 52.078 b 167.234 52.969 166.781 53.406 165.906 53.406 l 159.766 53.406 b 159.359 53.406 159.094 53.328 158.938 53.156 158.797 53 158.719 52.703 158.719 52.297 l 158.719 33.609 b 158.719 32.359 158.375 31.391 157.688 30.719 157.016 30.031 156.031 29.688 154.734 29.688 153.594 29.688 152.562 29.969 151.641 30.516 150.719 31.078 149.719 31.938 148.656 33.125 l 148.656 52.141 b 148.656 52.578 148.547 52.906 148.328 53.109 148.109 53.312 147.734 53.406 147.219 53.406 l 141.297 53.406 141.297 53.406'),
        Shape.from_ass_string('m 186.578 53.953 b 183.672 53.953 181.094 53.328 178.859 52.047 176.641 50.781 174.891 49 173.641 46.719 172.391 44.438 171.766 41.828 171.766 38.922 171.766 35.906 172.391 33.234 173.672 30.906 174.938 28.594 176.703 26.781 178.969 25.484 181.234 24.203 183.797 23.562 186.625 23.562 189.547 23.562 192.125 24.203 194.375 25.516 196.625 26.828 198.359 28.641 199.594 30.969 200.828 33.281 201.453 35.922 201.453 38.875 201.453 41.781 200.828 44.375 199.562 46.656 198.312 48.953 196.562 50.734 194.312 52.031 192.062 53.312 189.484 53.953 186.578 53.953 l 186.578 53.953'),
        Shape.from_ass_string('m 186.625 48.328 b 188.656 48.328 190.203 47.516 191.25 45.922 192.297 44.312 192.828 41.922 192.828 38.75 192.828 35.547 192.297 33.125 191.25 31.484 190.203 29.844 188.656 29.031 186.625 29.031 184.609 29.031 183.062 29.844 181.984 31.484 180.922 33.125 180.391 35.547 180.391 38.75 180.391 41.891 180.922 44.266 181.984 45.891 183.062 47.516 184.609 48.328 186.625 48.328 l 186.625 48.328')
    ]
    dest = Shape.from_ass_string('m 14.266 35.891 b 14.047 35.891 13.859 35.953 13.734 36.109 13.609 36.25 13.547 36.422 13.547 36.609 l 13.547 52.141 b 13.547 52.578 13.422 52.906 13.188 53.109 12.953 53.312 12.562 53.406 12.047 53.406 l 5.094 53.406 b 4.641 53.406 4.328 53.312 4.141 53.125 3.969 52.953 3.875 52.656 3.875 52.25 l 3.875 13.438 b 3.875 12.734 4.172 12.391 4.75 12.391 l 12.656 12.391 b 13.25 12.391 13.547 12.719 13.547 13.391 l 13.547 27.531 b 13.547 28.156 13.828 28.469 14.375 28.469 l 27.141 28.469 b 27.734 28.469 28.031 28.188 28.031 27.594 l 28.031 13.391 b 28.031 12.719 28.297 12.391 28.859 12.391 l 36.766 12.391 b 37.391 12.391 37.703 12.734 37.703 13.438 l 37.703 52.25 b 37.703 53.016 37.297 53.406 36.484 53.406 l 29.469 53.406 b 28.953 53.406 28.578 53.312 28.359 53.125 28.141 52.953 28.031 52.609 28.031 52.141 l 28.031 36.609 b  28.031 36.375 27.969 36.203 27.828 36.078 27.703 35.953 27.516 35.891 27.25 35.891 l 14.266 35.891 m 58.047 53.953 b 55.141 53.953 52.562 53.328 50.328 52.047 48.109 50.781 46.359 49 45.109 46.719 43.859 44.438 43.234 41.828 43.234 38.922 43.234 35.906 43.859 33.234 45.141 30.906 46.406 28.594 48.172 26.781 50.438 25.484 52.703 24.203 55.266 23.562 58.094 23.562 61.016 23.562 63.594 24.203 65.844 25.516 68.094 26.828 69.828 28.641 71.062 30.969 72.297 33.281 72.922 35.922 72.922 38.875 72.922 41.781 72.297 44.375 71.031 46.656 69.781 48.953 68.031 50.734 65.781 52.031 63.531 53.312 60.953 53.953 58.047 53.953 l 58.047 53.953 m 58.094 48.328 b 60.125 48.328 61.672 47.516 62.719 45.922 63.766 44.312 64.297 41.922 64.297 38.75 64.297 35.547 63.766 33.125 62.719 31.484 61.672 29.844 60.125 29.031 58.094 29.031 56.078 29.031 54.531 29.844 53.453 31.484 52.391 33.125 51.859 35.547 51.859 38.75 51.859 41.891 52.391 44.266 53.453 45.891 54.531 47.516 56.078 48.328 58.094 48.328 l 58.094 48.328 m 80.375 62.531 b 79.156 62.531 78.312 62.469 77.828 62.328 77.359 62.203 77.109 61.922 77.109 61.484 l 77.109 57.219 b 77.109 56.625 77.641 56.344 78.719 56.344 l 82.469 56.344 b 83.578 56.344 84.547 55.922 85.344 55.094 86.156 54.266 86.562 53.391 86.562 52.469 86.562 52.141 86.516 51.844 86.406 51.578 l 75.781 25.438 b 75.719 25.219 75.672 25.062 75.672 25 75.672 24.734 75.766 24.531 75.953 24.359 76.141 24.188 76.375 24.109 76.672 24.109 l 83.641 24.109 b 83.891 24.109 84.141 24.188 84.375 24.359 84.625 24.531 84.781 24.75 84.859 25.047 l 90.984 41.516 b 91.094 41.891 91.25 42.078 91.438 42.078 91.578 42.078 91.75 41.875 91.922 41.469 l 98.016 25  b 98.266 24.406 98.672 24.109 99.219 24.109 l 103.984 24.109 b 104.234 24.109 104.453 24.188 104.609 24.359 104.781 24.531 104.859 24.719 104.859 24.938 104.859 25.016 104.828 25.156 104.75 25.375 l 93.422 53.406 b 92.391 55.953 91.297 57.859 90.156 59.156 89.016 60.453 87.703 61.328 86.203 61.812 84.719 62.297 82.766 62.531 80.375 62.531 l 80.375 62.531 m 124.719 50.594 b 122.281 52.844 119.375 53.953 115.984 53.953 113.219 53.953 111.031 53.219 109.438 51.719 107.828 50.234 107.031 48.281 107.031 45.891 107.031 42.75 108.438 40.266 111.25 38.406 114.078 36.531 117.984 35.547 123 35.438 l 124.938 35.281 b 125.531 35.281 125.828 35.016 125.828 34.5 l 125.828 33.344 b 125.828 31.906 125.422 30.797 124.609 30.031 123.797 29.25 122.641 28.859 121.125 28.859 118.547 28.859 116.625 29.969 115.375 32.172 115.234 32.547 114.984 32.734 114.656 32.734 114.469 32.734 114.312 32.719 114.156 32.672 l 108.625 31.406 b 108.297 31.328 108.141 31.125 108.141 30.797 108.141 30.578 108.203 30.297 108.359 29.969 109.234 27.828 110.812 26.234 113.047 25.156 115.297 24.094 118.125 23.562 121.516 23.562 129.797 23.562 133.953 26.641 133.953 32.781 l 133.953 52.188 b 133.953 52.641 133.875 52.953 133.734 53.125 133.578 53.312 133.312 53.406 132.906 53.406 l 127.203 53.406 b 126.844 53.406 126.562 53.281 126.406 53.016 126.234 52.766 126.125 52.375 126.047 51.859 l 125.938 50.641 b 125.891 50.312 125.766 50.141 125.547 50.141 125.406 50.141 125.125 50.297 124.719 50.594 l 124.719 50.594 m 125.828 40.359 b 125.828 40.031 125.766 39.797 125.625 39.641 125.5 39.5 125.281 39.422 125 39.422 l 123.453 39.531 b 121.312 39.641 119.422 40.156 117.781 41.047 116.141 41.953 115.312 43.203 115.312 44.781 115.312 45.891 115.703 46.766 116.484 47.406 117.25 48.062 118.328 48.375 119.688 48.375 121.312 48.375 122.75 47.875 124 46.891 125.219 45.859 125.828 44.875 125.828 43.953 l 125.828 40.359 125.828 40.359 m 141.297 53.406 b 140.906 53.406 140.609 53.328 140.422 53.156 140.234 53 140.141 52.703 140.141 52.297 l 140.141 13.438 b 140.141 12.734 140.438 12.391 141.031 12.391 l  147.656 12.391 b 148.219 12.391 148.484 12.703 148.484 13.328 l 148.484 27.484 b 148.484 27.781 148.578 27.922 148.766 27.922 148.875 27.922 149 27.844 149.156 27.703 150.844 26.188 152.422 25.125 153.875 24.5 155.344 23.875 156.875 23.562 158.5 23.562 161.297 23.562 163.453 24.328 164.969 25.875 166.469 27.422 167.234 29.594 167.234 32.406 l 167.234 52.078 b 167.234 52.969 166.781 53.406 165.906 53.406 l 159.766 53.406 b 159.359 53.406 159.094 53.328 158.938 53.156 158.797 53 158.719 52.703 158.719 52.297 l 158.719 33.609 b 158.719 32.359 158.375 31.391 157.688 30.719 157.016 30.031 156.031 29.688 154.734 29.688 153.594 29.688 152.562 29.969 151.641 30.516 150.719 31.078 149.719 31.938 148.656 33.125 l 148.656 52.141 b 148.656 52.578 148.547 52.906 148.328 53.109 148.109 53.312 147.734 53.406 147.219 53.406 l 141.297  53.406 141.297 53.406 m 186.578 53.953 b 183.672 53.953 181.094 53.328 178.859 52.047 176.641 50.781 174.891 49 173.641 46.719 172.391 44.438 171.766 41.828 171.766 38.922 171.766 35.906 172.391 33.234 173.672 30.906 174.938 28.594 176.703 26.781 178.969 25.484 181.234 24.203 183.797 23.562 186.625 23.562 189.547 23.562 192.125 24.203 194.375 25.516 196.625 26.828 198.359 28.641 199.594 30.969 200.828 33.281 201.453 35.922 201.453 38.875 201.453 41.781 200.828 44.375 199.562 46.656 198.312 48.953 196.562 50.734 194.312 52.031 192.062 53.312 189.484 53.953 186.578 53.953 l 186.578 53.953 m 186.625 48.328 b 188.656 48.328 190.203 47.516 191.25 45.922 192.297 44.312 192.828 41.922 192.828 38.75 192.828 35.547 192.297 33.125 191.25 31.484 190.203 29.844 188.656 29.031 186.625 29.031 184.609 29.031 183.062 29.844 181.984 31.484 180.922 33.125 180.391 35.547 180.391 38.75 180.391 41.891 180.922 44.266 181.984 45.891 183.062 47.516 184.609 48.328 186.625 48.328 l 186.625 48.328')
    original = Shape.merge_shapes(originals)
    check.equal(original, dest)