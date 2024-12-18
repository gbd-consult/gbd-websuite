"""Tests for the bounds module."""
import math

import gws
import gws.test.util as u
import gws.lib.bounds as bounds
import gws.lib.crs


def test_from_request_bbox():
    bbox = '189000,834000,285000,962000,urn:x-ogc:def:crs:EPSG:4326'
    crs = gws.lib.crs.WGS84
    bound = gws.Bounds(crs=crs, extent=(834000.0, 189000.0, 962000.0, 285000.0))
    assert bounds.from_request_bbox(bbox).crs == bound.crs
    assert bounds.from_request_bbox(bbox).extent == bound.extent


def test_from_request_bbox_no_crs():
    bbox = '189000,834000,285000,962000'
    assert not bounds.from_request_bbox(bbox)


def test_from_request_bbox_default_crs():
    bbox = '189000,834000,285000,962000'
    crs = gws.lib.crs.WGS84
    bound = gws.Bounds(crs=crs, extent=(834000.0, 189000.0, 962000.0, 285000.0))
    assert bounds.from_request_bbox(bbox, crs).crs == bound.crs
    assert bounds.from_request_bbox(bbox, crs).extent == bound.extent


def test_from_request_bbox_empty():
    assert not bounds.from_request_bbox('')


def test_from_request_bbox_wrong_bbox():
    bbox = '189000,962000,urn:x-ogc:def:crs:EPSG:4326'
    assert not bounds.from_request_bbox(bbox)


def test_from_request_bbox_alwaysxy():
    bbox = '189000,834000,285000,962000'
    crs = gws.lib.crs.WGS84
    bound = gws.Bounds(crs=crs, extent=(189000.0, 834000.0, 285000.0, 962000.0))
    assert bounds.from_request_bbox(bbox, crs, always_xy=True).crs == bound.crs
    assert bounds.from_request_bbox(bbox, crs, always_xy=True).extent == bound.extent


def test_from_extent():
    extent = (100, 200, 300, 400)
    crs = gws.lib.crs.WGS84
    assert bounds.from_extent(extent, crs).extent == (200.0, 100.0, 400.0, 300.0)


def test_from_extent_alwaysxy():
    extent = (100, 200, 300, 400)
    crs = gws.lib.crs.WGS84
    assert bounds.from_extent(extent, crs, always_xy=True).extent == (100.0, 200.0, 300.0, 400.0)


def test_copy():
    crs = gws.lib.crs.WGS84
    extent = (100, 200, 300, 400)
    bound = gws.Bounds(crs=crs, extent=extent)
    assert not bounds.copy(bound) == bound
    assert bounds.copy(bound).crs == bound.crs
    assert bounds.copy(bound).extent == bound.extent


def test_union():
    wg = gws.lib.crs.WGS84
    web = gws.lib.crs.WEBMERCATOR
    bound1 = gws.Bounds(crs=wg, extent=(1, 100, 200, 200))
    bound2 = gws.Bounds(crs=web, extent=(100, 2, 200, 200))
    bound3 = gws.Bounds(crs=web, extent=(100, 100, 300, 200))
    bound4 = gws.Bounds(crs=web, extent=(100, 100, 200, 400))

    bound = gws.Bounds(crs=wg, extent=(0.0008983152841195213, 1.7966305682390134e-05, 200, 200))
    assert bounds.union([bound1, bound2, bound3, bound4]).extent == bound.extent
    assert bounds.union([bound1, bound2, bound3, bound4]).crs == bound.crs


def test_union_empty():
    with u.raises(Exception):
        bounds.union([])


def test_intersect():
    crs = gws.lib.crs.WEBMERCATOR
    b1 = gws.Bounds(crs=crs, extent=(100, 100, 400, 400))
    b2 = gws.Bounds(crs=crs, extent=(300, 300, 500, 500))

    assert bounds.intersect(b1, b2)


def test_intersect_empty():
    crs = gws.lib.crs.WEBMERCATOR
    b1 = gws.Bounds(crs=crs, extent=(100, 100, 400, 400))
    b2 = gws.Bounds(crs=crs, extent=(500, 500, 600, 600))

    assert not bounds.intersect(b1, b2)


def test_transform():
    crs = gws.lib.crs.WEBMERCATOR
    to_crs = gws.lib.crs.WGS84
    b1 = gws.Bounds(crs=crs, extent=(100, 100, 200, 200))
    b2 = gws.Bounds(crs=to_crs, extent=(8983, 8983, 17966, 17966))
    ext = bounds.transform(b1, to_crs).extent
    ext = (
        math.trunc(ext[0] * 10000000),
        math.trunc(ext[1] * 10000000),
        math.trunc(ext[2] * 10000000),
        math.trunc(ext[3] * 10000000)
    )
    assert bounds.transform(b1, to_crs).crs == b2.crs
    assert ext == b2.extent


def test_buffer():
    crs = gws.lib.crs.WGS84
    b = gws.Bounds(crs=crs, extent=(100, 100, 400, 400))
    assert bounds.buffer(b, 50).extent == (50, 50, 450, 450)
    assert bounds.buffer(b, 50).crs == crs
