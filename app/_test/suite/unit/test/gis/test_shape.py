import gws.gis.shape as shape
import shapely.geos as geos

import _test.util as u


def test_wkt():
    s = 'POINT (1000 2000)'
    p = shape.from_wkt(s, 'EPSG:3857')
    assert vars(p.props) == {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': (1000.0, 2000.0)}}
    assert p.wkt == s


def test_ewkt():
    s = 'SRID=25832;POINT (1000 2000)'
    p = shape.from_wkt(s)
    assert vars(p.props) == {'crs': 'EPSG:25832', 'geometry': {'type': 'Point', 'coordinates': (1000.0, 2000.0)}}
    assert p.ewkt == s


def test_wkb():
    # select st_geomfromtext('POINT (1000 2000)')
    h = '01010000000000000000408f400000000000409f40'
    b = bytes.fromhex(h)

    p = shape.from_wkb(b, 'EPSG:3857')
    assert vars(p.props) == {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': (1000.0, 2000.0)}}
    assert p.wkb == b

    p = shape.from_wkb_hex(h, 'EPSG:3857')
    assert vars(p.props) == {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': (1000.0, 2000.0)}}
    assert p.wkb_hex == h


def test_ewkb():
    # select st_asewkb(st_setsrid(st_geomfromtext('POINT (1000 2000)'),25832))
    h = '0101000020e86400000000000000408f400000000000409f40'
    b = bytes.fromhex(h)

    p = shape.from_wkb(b)
    assert vars(p.props) == {'crs': 'EPSG:25832', 'geometry': {'type': 'Point', 'coordinates': (1000.0, 2000.0)}}
    assert p.ewkb == b

    p = shape.from_wkb_hex(h)
    assert vars(p.props) == {'crs': 'EPSG:25832', 'geometry': {'type': 'Point', 'coordinates': (1000.0, 2000.0)}}
    assert p.ewkb_hex == h


def test_transform():
    a = 'SRID=3857;POLYGON ((800000 6000000, 800000 7000000, 700000 7000000, 700000 6000000, 800000 6000000))'
    # select st_asewkt(st_transform('SRID=3857;POLYGON ((800000 6000000, 800000 7000000, 700000 7000000, 700000 6000000, 800000 6000000))'::geometry,25832))
    # = 'SRID=25832;POLYGON((363043.661647077 5246065.74756768,378560.81939623 5884021.61510444,318415.744861985 5885921.80538396,295205.083254969 5248037.33419207,363043.661647077 5246065.74756768))'
    b = 'SRID=25832;POLYGON ((363043.662 5246065.748, 378560.819 5884021.615, 318415.745 5885921.805, 295205.083 5248037.334, 363043.662 5246065.748))'

    p = shape.from_wkt(a)
    q = shape.from_wkt(b)

    t = p.transformed(25832, precision=3)

    assert t.ewkt == b
