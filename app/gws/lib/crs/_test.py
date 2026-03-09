"""Tests for the crs module."""

import math

import gws
import gws.test.util as u
import gws.lib.crs as crs


# --- get / parse / require ---


def test_get_no_name():
    assert not crs.get('')


def test_get_none():
    assert crs.get(None) is None


def test_get_unknown():
    assert crs.get('EPSG:0') is None
    assert crs.get('FOOBAR') is None


def test_get_by_epsg():
    assert str(crs.get('EPSG:3857')) == '<crs:3857>'
    assert crs.get('EPSG:3857') == crs.WEBMERCATOR


def test_get_by_srid():
    assert crs.get('3857') == crs.WEBMERCATOR
    assert crs.get('4326') == crs.WGS84


def test_get_by_int():
    assert crs.get(3857) == crs.WEBMERCATOR
    assert crs.get(4326) == crs.WGS84


def test_get_dynamic_crs():
    # UTM zone 33N - not predefined, loaded dynamically
    c = crs.get('EPSG:25833')
    assert c is not None
    assert c.srid == 25833
    assert c.isProjected is True
    assert c.uom == gws.Uom.m


def test_get_dynamic_crs_cached():
    c1 = crs.get('EPSG:25833')
    c2 = crs.get('25833')
    assert c1 is c2


def test_parse():
    fmt, obj = crs.parse('EPSG:3857')
    assert fmt == gws.CrsFormat.epsg
    assert obj == crs.get('3857')


def test_parse_srid():
    fmt, obj = crs.parse('3857')
    assert fmt == gws.CrsFormat.srid
    assert obj == crs.get('3857')


def test_parse_urn():
    fmt, obj = crs.parse('urn:ogc:def:crs:EPSG::4326')
    assert fmt == gws.CrsFormat.urn
    assert obj == crs.WGS84


def test_parse_url():
    fmt, obj = crs.parse('http://www.opengis.net/gml/srs/epsg.xml#3857')
    assert fmt == gws.CrsFormat.url
    assert obj == crs.WEBMERCATOR


def test_parse_uri():
    fmt, obj = crs.parse('http://www.opengis.net/def/crs/epsg/0/4326')
    assert fmt == gws.CrsFormat.uri
    assert obj == crs.WGS84


def test_parse_aliases():
    fmt, obj = crs.parse('CRS:84')
    assert fmt == gws.CrsFormat.crs
    assert obj == crs.WGS84

    fmt, obj = crs.parse('epsg:900913')
    assert fmt == gws.CrsFormat.epsg
    assert obj == crs.WEBMERCATOR

    fmt, obj = crs.parse('epsg:102100')
    assert fmt == gws.CrsFormat.epsg
    assert obj == crs.WEBMERCATOR


def test_parse_no_format():
    fmt, obj = crs.parse('11wrongFormat')
    assert fmt == gws.CrsFormat.none
    assert obj is None


def test_require():
    assert crs.require('3857') == crs.get('3857')


def test_require_exception():
    with u.raises(crs.Error):
        crs.require('FOOBAR')


# --- CRS object properties ---


def test_crs_equality():
    a = crs.get('EPSG:3857')
    b = crs.get('3857')
    assert a == b


def test_crs_hash():
    a = crs.get('EPSG:3857')
    b = crs.get('3857')
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_crs_repr():
    assert repr(crs.WGS84) == '<crs:4326>'
    assert repr(crs.WEBMERCATOR) == '<crs:3857>'


def test_wgs84_properties():
    assert crs.WGS84.srid == 4326
    assert crs.WGS84.isGeographic is True
    assert crs.WGS84.isProjected is False
    assert crs.WGS84.uom == gws.Uom.deg
    assert crs.WGS84.axis == gws.Axis.yx
    assert crs.WGS84.isYX is True


def test_webmercator_properties():
    assert crs.WEBMERCATOR.srid == 3857
    assert crs.WEBMERCATOR.isGeographic is False
    assert crs.WEBMERCATOR.isProjected is True
    assert crs.WEBMERCATOR.uom == gws.Uom.m
    assert crs.WEBMERCATOR.axis == gws.Axis.xy
    assert crs.WEBMERCATOR.isYX is False


def test_wgs84_extent():
    assert crs.WGS84.extent == (-180, -90, 180, 90)
    assert crs.WGS84.wgsExtent == (-180, -90, 180, 90)


def test_webmercator_extent():
    e = crs.WEBMERCATOR.extent
    assert e[0] < -20_000_000
    assert e[2] > 20_000_000


def test_webmercator_square():
    r = math.pi * crs.WEBMERCATOR_RADIUS
    assert u.is_close(crs.WEBMERCATOR_SQUARE, (-r, -r, r, r))


# --- best_match ---


def test_best_match():
    lst = [crs.WEBMERCATOR, crs.WGS84]
    assert crs.best_match(crs.WGS84, lst) == crs.get('4326')


def test_best_match_not_in_list():
    lst = [crs.WEBMERCATOR]
    assert crs.best_match(crs.WGS84, lst) == crs.get('3857')


def test_best_match_projected_prefers_webmercator():
    c = crs.get('EPSG:25833')
    lst = [crs.WGS84, crs.WEBMERCATOR]
    assert crs.best_match(c, lst) == crs.WEBMERCATOR


def test_best_match_geographic_prefers_wgs84():
    lst = [crs.WEBMERCATOR, crs.WGS84]
    assert crs.best_match(crs.WGS84, lst) == crs.WGS84


def test_best_match_exact():
    lst = [crs.WEBMERCATOR, crs.WGS84]
    assert crs.best_match(crs.WEBMERCATOR, lst) == crs.WEBMERCATOR


# --- axis_for_format ---


def test_axis_for_format():
    # WGS84 is internally YX, but output axis depends on the requested CRS string format.
    assert crs.WGS84.axis_for_format(gws.CrsFormat.epsg) == gws.Axis.xy
    assert crs.WGS84.axis_for_format(gws.CrsFormat.urn) == gws.Axis.yx


def test_axis_for_format_webmercator():
    # WebMercator is XY natively, so all formats are XY
    assert crs.WEBMERCATOR.axis_for_format(gws.CrsFormat.epsg) == gws.Axis.xy
    assert crs.WEBMERCATOR.axis_for_format(gws.CrsFormat.urn) == gws.Axis.xy


def test_axis_for_format_all_wgs84():
    assert crs.WGS84.axis_for_format(gws.CrsFormat.srid) == gws.Axis.xy
    assert crs.WGS84.axis_for_format(gws.CrsFormat.epsg) == gws.Axis.xy
    assert crs.WGS84.axis_for_format(gws.CrsFormat.url) == gws.Axis.xy
    assert crs.WGS84.axis_for_format(gws.CrsFormat.uri) == gws.Axis.xy
    assert crs.WGS84.axis_for_format(gws.CrsFormat.urnx) == gws.Axis.yx
    assert crs.WGS84.axis_for_format(gws.CrsFormat.urn) == gws.Axis.yx


# --- to_string ---


def test_to_string():
    assert crs.WEBMERCATOR.to_string() == 'EPSG:3857'
    assert crs.WEBMERCATOR.to_string(gws.CrsFormat.epsg) == 'EPSG:3857'
    assert crs.WEBMERCATOR.to_string(gws.CrsFormat.srid) == '3857'


def test_to_string_urn():
    assert crs.WGS84.to_string(gws.CrsFormat.urn) == 'urn:ogc:def:crs:EPSG::4326'


def test_to_string_url():
    assert crs.WGS84.to_string(gws.CrsFormat.url) == 'http://www.opengis.net/gml/srs/epsg.xml#4326'


def test_to_string_uri():
    assert crs.WGS84.to_string(gws.CrsFormat.uri) == 'http://www.opengis.net/def/crs/epsg/0/4326'


# --- to_geojson ---


def test_to_geojson():
    assert crs.WEBMERCATOR.to_geojson() == {
        'properties': {'name': 'urn:ogc:def:crs:EPSG::3857'},
        'type': 'name',
    }


def test_to_geojson_wgs84():
    assert crs.WGS84.to_geojson() == {
        'properties': {'name': 'urn:ogc:def:crs:EPSG::4326'},
        'type': 'name',
    }


# --- transform_extent ---


def test_transform_extent_wgs84_to_webmercator():
    got = crs.WGS84.transform_extent((0.0, 1.0, 1.0, 0.0), crs_to=crs.WEBMERCATOR)
    exp = (0.0, 0.0, 111319.49079327357, 111325.1428663851)
    assert u.is_close(got, exp, abs_tol=1e-4)


def test_transform_extent_webmercator_to_wgs84():
    ext = (0.0, 0.0, 111319.49079327357, 111325.1428663851)
    got = crs.WEBMERCATOR.transform_extent(ext, crs_to=crs.WGS84)
    exp = (0.0, 0.0, 1.0, 1.0)
    assert u.is_close(got, exp, abs_tol=1e-4)


def test_transform_extent_same_crs():
    ext = (10.0, 20.0, 30.0, 40.0)
    got = crs.WGS84.transform_extent(ext, crs_to=crs.WGS84)
    assert got == ext


def test_transform_extent_normalizes():
    # Swapped min/max should be normalized
    got = crs.WGS84.transform_extent((1.0, 1.0, 0.0, 0.0), crs_to=crs.WEBMERCATOR)
    exp = (0.0, 0.0, 111319.49079327357, 111325.1428663851)
    assert u.is_close(got, exp, abs_tol=1e-4)


def test_transform_extent_wgs84_to_utm():
    # A small area in the UTM 33N zone (lon 12-18)
    c33 = crs.get('EPSG:25833')
    got = crs.WGS84.transform_extent((12.0, 50.0, 13.0, 51.0), crs_to=c33)
    # Result should be in meters, reasonable UTM values
    assert got[0] > 0  # easting > 0
    assert got[1] > 5_000_000  # northing > 5M (for lat ~50)
    assert got[2] > got[0]
    assert got[3] > got[1]


def test_transform_extent_utm_to_wgs84():
    c33 = crs.get('EPSG:25833')
    # Roughly the center of UTM zone 33
    ext = (300_000, 5_500_000, 400_000, 5_600_000)
    got = c33.transform_extent(ext, crs_to=crs.WGS84)
    # Should be in the right area (lon 12-18, lat ~49-51)
    assert 10 < got[0] < 20
    assert 45 < got[1] < 55
    assert 10 < got[2] < 20
    assert 45 < got[3] < 55


def test_transform_extent_utm_to_webmercator():
    c33 = crs.get('EPSG:25833')
    ext = (300_000, 5_500_000, 400_000, 5_600_000)
    got = c33.transform_extent(ext, crs_to=crs.WEBMERCATOR)
    # Should be in WebMercator meters
    assert got[0] > 1_000_000
    assert got[2] > got[0]
    assert got[3] > got[1]


def test_transform_extent_webmercator_to_utm():
    c33 = crs.get('EPSG:25833')
    # Area within UTM 33 bounds in WebMercator
    ext = crs.WGS84.transform_extent((13.0, 50.0, 14.0, 51.0), crs_to=crs.WEBMERCATOR)
    got = crs.WEBMERCATOR.transform_extent(ext, crs_to=c33)
    # Should be reasonable UTM coordinates
    assert got[0] > 0
    assert got[1] > 5_000_000


# --- transform_extent: big extent to narrow projection ---


def test_transform_extent_world_wgs84_to_utm():
    """Transforming WGS84 world extent to UTM should not fail and should produce a wide extent."""
    c33 = crs.get('EPSG:25833')
    got = crs.WGS84.transform_extent((-180.0, -85.06, 180.0, 85.06), crs_to=c33)
    # Should not crash, and should have a wide X range (not a narrow band)
    assert got is not None
    x_range = got[2] - got[0]
    y_range = got[3] - got[1]
    # X range should be much wider than the native UTM zone (~600km)
    assert x_range > 5_000_000, f'X range too narrow: {x_range}'
    assert y_range > 10_000_000, f'Y range too narrow: {y_range}'


def test_transform_extent_world_webmercator_to_utm():
    """Transforming full WebMercator extent to UTM should not fail."""
    c33 = crs.get('EPSG:25833')
    got = crs.WEBMERCATOR.transform_extent(crs.WEBMERCATOR.extent, crs_to=c33)
    assert got is not None
    x_range = got[2] - got[0]
    y_range = got[3] - got[1]
    assert x_range > 5_000_000, f'X range too narrow: {x_range}'
    assert y_range > 10_000_000, f'Y range too narrow: {y_range}'


def test_transform_extent_full_wgs84_to_webmercator():
    """Transforming full WGS84 to WebMercator should work normally (both global)."""
    got = crs.WGS84.transform_extent((-180, -85, 180, 85), crs_to=crs.WEBMERCATOR)
    assert got[0] < -20_000_000
    assert got[2] > 20_000_000


def test_transform_extent_big_symmetric_y():
    """When transforming world to UTM, Y range should be roughly symmetric around zero or a central value."""
    c33 = crs.get('EPSG:25833')
    got = crs.WGS84.transform_extent((-180.0, -85.06, 180.0, 85.06), crs_to=c33)
    # The Y should have large negative and large positive values
    assert got[1] < -1_000_000, f'Y min not negative enough: {got[1]}'
    assert got[3] > 1_000_000, f'Y max not positive enough: {got[3]}'


def test_transform_extent_big_to_other_utm_zones():
    """Big extent transform should work for different UTM zones."""
    world = (-180.0, -85.06, 180.0, 85.06)
    for srid in [25832, 25833, 25834]:  # UTM zones 32, 33, 34
        c = crs.get(f'EPSG:{srid}')
        got = crs.WGS84.transform_extent(world, crs_to=c)
        assert got is not None
        x_range = got[2] - got[0]
        assert x_range > 5_000_000, f'EPSG:{srid} X range too narrow: {x_range}'


# --- transform_extent: tmerc (Transverse Mercator) ---


def test_transform_extent_wgs84_to_british_national_grid():
    """EPSG:27700 British National Grid (tmerc), small area within its AoU."""
    bng = crs.get('EPSG:27700')
    # London area (lon ~-0.1, lat ~51.5)
    got = crs.WGS84.transform_extent((-1.0, 51.0, 0.0, 52.0), crs_to=bng)
    assert got[0] < got[2]
    assert got[1] < got[3]
    # BNG easting for London area should be ~500km
    assert 400_000 < got[0] < 600_000
    assert 400_000 < got[2] < 700_000


def test_transform_extent_british_national_grid_to_wgs84():
    """Roundtrip: BNG -> WGS84."""
    bng = crs.get('EPSG:27700')
    ext = (400_000, 100_000, 600_000, 400_000)
    got = bng.transform_extent(ext, crs_to=crs.WGS84)
    # Should land in the UK area
    assert -10 < got[0] < 5
    assert 49 < got[1] < 62
    assert -10 < got[2] < 5
    assert 49 < got[3] < 62


def test_transform_extent_world_to_british_national_grid():
    """Transforming world extent to BNG (tmerc) should produce a wide extent, not a narrow band."""
    bng = crs.get('EPSG:27700')
    world = (-180.0, -85.06, 180.0, 85.06)
    got = crs.WGS84.transform_extent(world, crs_to=bng)
    assert got is not None
    x_range = got[2] - got[0]
    y_range = got[3] - got[1]
    # Should be much wider than the native BNG area (~700km x ~1300km)
    assert x_range > 5_000_000, f'BNG X range too narrow: {x_range}'
    assert y_range > 10_000_000, f'BNG Y range too narrow: {y_range}'


def test_transform_extent_webmercator_to_british_national_grid():
    """Full WebMercator to BNG (tmerc) should not fail."""
    bng = crs.get('EPSG:27700')
    got = crs.WEBMERCATOR.transform_extent(crs.WEBMERCATOR.extent, crs_to=bng)
    assert got is not None
    x_range = got[2] - got[0]
    assert x_range > 5_000_000, f'BNG X range too narrow: {x_range}'


def test_transform_extent_wgs84_to_utm32n():
    """EPSG:32632 UTM zone 32N (tmerc), small area within its AoU."""
    utm32 = crs.get('EPSG:32632')
    # Area in Germany (lon 6-12, lat ~50)
    got = crs.WGS84.transform_extent((8.0, 49.0, 10.0, 51.0), crs_to=utm32)
    assert got[0] > 0
    assert got[1] > 5_000_000
    assert got[2] > got[0]
    assert got[3] > got[1]


def test_transform_extent_world_to_utm32n():
    """World extent to UTM 32N (tmerc) should produce a wide extent."""
    utm32 = crs.get('EPSG:32632')
    world = (-180.0, -85.06, 180.0, 85.06)
    got = crs.WGS84.transform_extent(world, crs_to=utm32)
    assert got is not None
    x_range = got[2] - got[0]
    assert x_range > 5_000_000, f'UTM32N X range too narrow: {x_range}'


# --- transform_extent: lcc (Lambert Conformal Conic) ---


def test_transform_extent_wgs84_to_lambert93():
    """EPSG:2154 Lambert-93 France (lcc), small area within its AoU."""
    lcc = crs.get('EPSG:2154')
    # Paris area (lon ~2.3, lat ~48.9)
    got = crs.WGS84.transform_extent((2.0, 48.0, 3.0, 49.0), crs_to=lcc)
    assert got[0] < got[2]
    assert got[1] < got[3]
    # Lambert-93 easting for Paris ≈ 650km, northing ≈ 6850km
    assert 500_000 < got[0] < 800_000
    assert 6_700_000 < got[1] < 7_000_000


def test_transform_extent_lambert93_to_wgs84():
    """Roundtrip: Lambert-93 -> WGS84."""
    lcc = crs.get('EPSG:2154')
    ext = (100_000, 6_100_000, 1_000_000, 7_100_000)
    got = lcc.transform_extent(ext, crs_to=crs.WGS84)
    # Should land in France area
    assert -10 < got[0] < 15
    assert 40 < got[1] < 55
    assert -10 < got[2] < 15
    assert 40 < got[3] < 55


def test_transform_extent_world_to_lambert93():
    """Transforming world extent to Lambert-93 (lcc) should produce a wide extent."""
    lcc = crs.get('EPSG:2154')
    world = (-180.0, -85.06, 180.0, 85.06)
    got = crs.WGS84.transform_extent(world, crs_to=lcc)
    assert got is not None
    x_range = got[2] - got[0]
    y_range = got[3] - got[1]
    # Should be much wider than the native Lambert-93 area (~1100km x ~1200km)
    assert x_range > 5_000_000, f'Lambert-93 X range too narrow: {x_range}'
    assert y_range > 10_000_000, f'Lambert-93 Y range too narrow: {y_range}'


def test_transform_extent_webmercator_to_lambert93():
    """Full WebMercator to Lambert-93 (lcc) should not fail."""
    lcc = crs.get('EPSG:2154')
    got = crs.WEBMERCATOR.transform_extent(crs.WEBMERCATOR.extent, crs_to=lcc)
    assert got is not None
    x_range = got[2] - got[0]
    assert x_range > 5_000_000, f'Lambert-93 X range too narrow: {x_range}'


def test_transform_extent_wgs84_to_laea_europe():
    """EPSG:3035 Lambert Azimuthal Equal Area Europe (laea), broader coverage."""
    laea = crs.get('EPSG:3035')
    # Berlin area
    got = crs.WGS84.transform_extent((13.0, 52.0, 14.0, 53.0), crs_to=laea)
    assert got[0] < got[2]
    assert got[1] < got[3]
    # LAEA Europe easting ~4500km, northing ~3300km for Berlin
    assert 4_000_000 < got[0] < 5_000_000
    assert 3_000_000 < got[1] < 4_000_000


def test_transform_extent_world_to_laea_europe():
    """World extent to LAEA Europe should produce a wide extent."""
    laea = crs.get('EPSG:3035')
    world = (-180.0, -85.06, 180.0, 85.06)
    got = crs.WGS84.transform_extent(world, crs_to=laea)
    assert got is not None
    x_range = got[2] - got[0]
    y_range = got[3] - got[1]
    assert x_range > 5_000_000, f'LAEA Europe X range too narrow: {x_range}'
    assert y_range > 5_000_000, f'LAEA Europe Y range too narrow: {y_range}'


# --- cross-projection transforms ---


def test_transform_extent_bng_to_lambert93():
    """Transform between two narrow projections (tmerc -> lcc)."""
    bng = crs.get('EPSG:27700')
    lcc = crs.get('EPSG:2154')
    # London area in BNG
    ext_bng = (500_000, 100_000, 600_000, 200_000)
    got = bng.transform_extent(ext_bng, crs_to=lcc)
    assert got[0] < got[2]
    assert got[1] < got[3]


def test_transform_extent_utm_to_lambert93():
    """Transform between UTM 33 (tmerc) and Lambert-93 (lcc)."""
    utm33 = crs.get('EPSG:25833')
    lcc = crs.get('EPSG:2154')
    # Area in eastern Germany/western Poland, within both CRS areas
    ext_utm = (300_000, 5_500_000, 400_000, 5_600_000)
    got = utm33.transform_extent(ext_utm, crs_to=lcc)
    assert got[0] < got[2]
    assert got[1] < got[3]
    # Should be in the eastern part of Lambert-93 coverage
    assert got[0] > 500_000


# --- _is_big_extent ---


def test_is_big_extent_world():
    assert crs._is_big_extent((-180, -90, 180, 90)) is True


def test_is_big_extent_webmercator_wgs():
    assert crs._is_big_extent((-180, -85.06, 180, 85.06)) is True


def test_is_big_extent_small():
    assert crs._is_big_extent((12, 50, 13, 51)) is False


def test_is_big_extent_wide_lon_only():
    # Wide longitude but narrow latitude
    assert crs._is_big_extent((-180, 0, 180, 10)) is True


def test_is_big_extent_tall_lat_only():
    # Narrow longitude but tall latitude
    assert crs._is_big_extent((0, -90, 10, 90)) is True


# --- transformer ---


def test_transformer():
    tr = crs.WGS84.transformer(crs.WEBMERCATOR)
    assert callable(tr)
    x, y = tr(1.0, 1.0)  # lon/lat -> meters (always_xy=True)
    assert u.is_close(x, 111319.49079327357, abs_tol=1e-4)
    assert u.is_close(y, 111325.1428663851, abs_tol=1e-4)


def test_transformer_roundtrip():
    tr_fwd = crs.WGS84.transformer(crs.WEBMERCATOR)
    tr_inv = crs.WEBMERCATOR.transformer(crs.WGS84)
    x, y = tr_fwd(10.0, 50.0)
    lon, lat = tr_inv(x, y)
    assert u.is_close(lon, 10.0, abs_tol=1e-6)
    assert u.is_close(lat, 50.0, abs_tol=1e-6)


def test_transformer_utm():
    c33 = crs.get('EPSG:25833')
    tr = crs.WGS84.transformer(c33)
    x, y = tr(15.0, 51.0)  # center of zone 33
    assert 400_000 < x < 600_000
    assert 5_600_000 < y < 5_700_000


# --- extent_size_in_meters ---


def test_extent_size_in_meters_projected():
    # 1000m x 2000m rectangle in WEBMERCATOR
    ext = (0, 0, 1000, 2000)
    w, h = crs.WEBMERCATOR.extent_size_in_meters(ext)
    assert u.is_close(w, 1000, abs_tol=1e-6)
    assert u.is_close(h, 2000, abs_tol=1e-6)


def test_extent_size_in_meters_geographic():
    # 1 degree longitude at equator ≈ 111319.5m, 1 degree latitude ≈ 110574m
    ext = (0, 0, 1, 1)
    w, h = crs.WGS84.extent_size_in_meters(ext)
    assert u.is_close(w, 111319.5, rel_tol=1e-3)
    assert u.is_close(h, 110574, rel_tol=1e-3)


def test_extent_size_in_meters_utm():
    c33 = crs.get('EPSG:25833')
    ext = (500_000, 5_500_000, 501_000, 5_502_000)
    w, h = c33.extent_size_in_meters(ext)
    assert u.is_close(w, 1000, abs_tol=1e-6)
    assert u.is_close(h, 2000, abs_tol=1e-6)


# --- point_offset_in_meters ---


def test_point_offset_north():
    x, y = crs.WEBMERCATOR.point_offset_in_meters((500_000, 5_500_000), 1000, 0)
    assert u.is_close(x, 500_000)
    assert u.is_close(y, 5_501_000)


def test_point_offset_east():
    x, y = crs.WEBMERCATOR.point_offset_in_meters((500_000, 5_500_000), 1000, 90)
    assert u.is_close(x, 501_000)
    assert u.is_close(y, 5_500_000)


def test_point_offset_south():
    x, y = crs.WEBMERCATOR.point_offset_in_meters((500_000, 5_500_000), 1000, 180)
    assert u.is_close(x, 500_000)
    assert u.is_close(y, 5_499_000)


def test_point_offset_west():
    x, y = crs.WEBMERCATOR.point_offset_in_meters((500_000, 5_500_000), 1000, 270)
    assert u.is_close(x, 499_000)
    assert u.is_close(y, 5_500_000)


def test_point_offset_diagonal():
    x, y = crs.WEBMERCATOR.point_offset_in_meters((0, 0), 1000, 45)
    assert x > 0
    assert y > 0
    dist = math.sqrt(x ** 2 + y ** 2)
    assert u.is_close(dist, 1000, abs_tol=1e-6)


def test_point_offset_geographic():
    x, y = crs.WGS84.point_offset_in_meters((0, 0), 111319.5, 90)
    # 111319.5m east at equator ≈ 1 degree longitude
    assert u.is_close(x, 1.0, abs_tol=0.01)
    assert u.is_close(y, 0.0, abs_tol=0.01)


# --- _normalize_extent ---


def test_normalize_extent_already_normal():
    assert crs._normalize_extent((0, 0, 10, 10)) == (0, 0, 10, 10)


def test_normalize_extent_swapped():
    assert crs._normalize_extent((10, 10, 0, 0)) == (0, 0, 10, 10)


def test_normalize_extent_mixed():
    assert crs._normalize_extent((10, 0, 0, 10)) == (0, 0, 10, 10)


# --- dynamic CRS properties ---


def test_dynamic_crs_extent():
    c33 = crs.get('EPSG:25833')
    e = c33.extent
    # UTM 33N extent should be reasonable
    assert e[0] < e[2]
    assert e[1] < e[3]
    assert e[0] > 100_000
    assert e[2] < 1_000_000


def test_dynamic_crs_wgs_extent():
    c33 = crs.get('EPSG:25833')
    we = c33.wgsExtent
    # Should be within UTM zone 33 area of use (lon 12-18, lat up to ~84)
    assert 10 < we[0] < 20
    assert 0 < we[1] < 90
    assert 10 < we[2] < 20
    assert 0 < we[3] < 90


def test_dynamic_crs_bounds():
    c33 = crs.get('EPSG:25833')
    assert c33.bounds is not None
    assert c33.bounds.crs == c33
    assert c33.bounds.extent == c33.extent


# --- qgis_extent_width ---


def test_qgis_extent_width():
    # 1 degree at equator
    w = crs.qgis_extent_width((0, 0, 1, 0))
    assert w > 100_000  # should be ~111km
    assert w < 120_000
