"""Tests for the extent module."""
import math

import gws
import gws.test.util as u
import gws.gis.extent as extent

import gws.gis.crs as crs


def test_from_string():
    assert extent.from_string('100,200,200,400') == (100, 200, 200, 400)


def test_from_list():
    assert extent.from_list([100, 200, 300, 400]) == (100, 200, 300, 400)


def test_from_list_error():
    assert not extent.from_list([100, 200])


def test_from_list_value_error():
    assert not extent.from_list(['a', 'b', 100, 200])


def test_from_list_same_point():
    assert not extent.from_list([100, 200, 100, 200])


def test_from_points():
    a = (100.0, 200.0)
    b = (300.0, 400.0)
    assert extent.from_points(a, b) == (100, 200, 300, 400)


def test_from_center():
    a = (100.0, 200.0)
    size = (50, 100)
    assert extent.from_center(a, size) == (75, 150, 125, 250)


def test_from_box():
    assert extent.from_box('box(100 200,300 400)') == (100, 200, 300, 400)


def test_from_box_commas():
    with u.raises(Exception):
        extent.from_box('box(100,200,300,400)')


def test_from_box_pattern():
    assert not extent.from_box('foo(100 200, 300 400)')


def test_from_box_empty():
    assert not extent.from_box('')


def test_intersection():
    a = (100, 100, 300, 300)
    b = (200, 200, 400, 400)
    c = (200, 100, 400, 300)
    exts = [a, b, c]
    assert extent.intersection(exts) == (200, 200, 300, 300)


def test_intersection_empty():
    a = (100, 100, 300, 300)
    b = (200, 200, 400, 400)
    c = (500, 600, 700, 700)
    exts = [a, b, c]
    assert not extent.intersection(exts)


def test_intersection_empty_list():
    assert not extent.intersection([])


def test_center():
    assert extent.center((100, 100, 200, 200)) == (150, 150)


def test_size():
    assert extent.size((100, 100, 200, 200)) == (100, 100)


def test_diagonal():
    assert extent.diagonal((1, 1, 4, 5)) == 5


def test_circumsquare():
    assert extent.circumsquare((1, 1, 4, 5)) == (0, 0.5, 5, 5.5)


def test_buffer():
    assert extent.buffer((100, 100, 200, 200), 100) == (0, 0, 300, 300)


def test_union():
    exts = [
        (1, 100, 200, 200),
        (100, 2, 200, 200),
        (100, 100, 300, 200),
        (100, 100, 200, 400)
    ]
    assert extent.union(exts) == (1, 2, 300, 400)


def test_union_empty():
    with u.raises(Exception):
        extent.union([])


def test_intersect():
    a = (300, 400, 700, 800)
    b = (100, 200, 500, 600)
    assert extent.intersect(a, b)


def test_intersect_inf():
    a = (1, 2, 3, 4)
    b = (-math.inf, -math.inf, math.inf, math.inf)
    assert extent.intersect(a, b)


def test_intersect_false():
    a = (300, 300, 400, 400)
    b = (100, 100, 200, 200)
    assert not extent.intersect(a, b)


def test_transform():
    from_crs = crs.WEBMERCATOR
    to_crs = crs.WGS84
    ext = extent.transform((100, 100, 200, 200), from_crs, to_crs)
    ext = (
        math.trunc(ext[0] * 10000000),
        math.trunc(ext[1] * 10000000),
        math.trunc(ext[2] * 10000000),
        math.trunc(ext[3] * 10000000)
    )
    assert ext == (8983, 8983, 17966, 17966)


def test_transform_to_wgs():
    from_crs = crs.WEBMERCATOR
    ext = extent.transform_to_wgs((100, 100, 200, 200), from_crs)
    ext = (
        math.trunc(ext[0] * 10000000),
        math.trunc(ext[1] * 10000000),
        math.trunc(ext[2] * 10000000),
        math.trunc(ext[3] * 10000000)
    )
    assert ext == (8983, 8983, 17966, 17966)


def test_transform_from_wgs():
    to_crs = crs.WEBMERCATOR
    ext = extent.transform_from_wgs((0.0008983, 0.0008983, 0.0017967, 0.0017967), to_crs)
    ext = (
        math.trunc(ext[0]),
        math.trunc(ext[1]),
        math.trunc(ext[2]),
        math.trunc(ext[3])
    )
    assert ext == (99, 99, 200, 200)


def test_swap_xy():
    assert extent.swap_xy((2, 1, 4, 3)) == (1, 2, 3, 4)


def test_is_valid():
    assert extent.is_valid([1, 1, 2, 2])
    assert extent.is_valid([1.123, 1.123, 2.123, 2.123])

    assert not extent.is_valid([1, 1, 1, 1])
    assert not extent.is_valid([2.2, 1, 1])
    assert not extent.is_valid([1, 2, 3, 4, 5])
    assert not extent.is_valid([])
    assert not extent.is_valid([1, 2])
    assert not extent.is_valid(None)
    assert not extent.is_valid([1, 2, 3, math.inf])
    assert not extent.is_valid([float("nan")] * 4)
