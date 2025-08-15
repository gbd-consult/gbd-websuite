import gws
import gws.test.util as u
import gws.base.shape as shape
import gws.lib.crs as crs
import shapely.wkt
import shapely.wkb
import geoalchemy2.shape


def test_from_wkt():
    default_crs = crs.WGS84
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    assert shape.from_wkt(wkt, default_crs).geom == shapely.wkt.loads(wkt)
    assert shape.from_wkt(wkt, default_crs).crs == default_crs
    assert shape.from_wkt(wkt, default_crs).type == 'polygon'
    assert not shape.from_wkt(wkt, default_crs).x
    assert not shape.from_wkt(wkt, default_crs).y


def test_from_wkt_raises():
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    default_crs = None
    with u.raises(Exception):
        shape.from_wkt(wkt, default_crs)


def test_from_ewkt():
    ewkt = 'SRID=4326;POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    assert shape.from_wkt(ewkt).geom == shapely.wkt.loads(wkt)
    assert shape.from_wkt(ewkt).crs.epsg == 'EPSG:4326'
    assert shape.from_wkt(ewkt).type == 'polygon'
    assert not shape.from_wkt(ewkt).x
    assert not shape.from_wkt(ewkt).y


def test_from_wkb():
    default_crs = crs.WGS84
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    wkb = shapely.wkb.dumps(shapely.from_wkt(wkt))
    assert shape.from_wkb(wkb, default_crs).geom == shapely.wkb.loads(wkb)
    assert shape.from_wkb(wkb, default_crs).crs == default_crs
    assert shape.from_wkb(wkb, default_crs).type == 'polygon'
    assert not shape.from_wkb(wkb, default_crs).x
    assert not shape.from_wkb(wkb, default_crs).y


def test_from_wkb_raises():
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    wkb = shapely.wkb.dumps(shapely.from_wkt(wkt))
    default_crs = None
    with u.raises(Exception):
        shape.from_wkb(wkb, default_crs)


def test_from_ewkb():
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    ewkb = shapely.wkb.dumps(shapely.from_wkt(wkt), srid=4326)
    assert shape.from_wkb(ewkb).geom == shapely.wkb.loads(ewkb)
    assert shape.from_wkb(ewkb).crs.epsg == 'EPSG:4326'
    assert shape.from_wkb(ewkb).type == 'polygon'
    assert not shape.from_wkb(ewkb).x
    assert not shape.from_wkb(ewkb).y


WKB_HEX = (
    "0103000000010000000500000000000000000000000000000000000000000000000000F03F000000000000F03F"
    "000000000000F03F00000000000000000000000000000000000000000000"
)
EWKB_HEX = (
    "0103000020E6100000010000000500000000000000000000000000000000000000000000000000F03F00000000"
    "0000F03F000000000000F03F00000000000000000000000000000000000000000000"
)


def test_from_wkb_hex():
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    wkb_hex = shapely.wkb.dumps(shapely.from_wkt(wkt), hex=True)
    default_crs = crs.WGS84
    assert shape.from_wkb_hex(wkb_hex, default_crs).geom == shapely.wkb.loads(bytes.fromhex(wkb_hex))
    assert shape.from_wkb_hex(wkb_hex, default_crs).crs == default_crs
    assert shape.from_wkb_hex(wkb_hex, default_crs).type == 'polygon'
    assert not shape.from_wkb_hex(wkb_hex, default_crs).x
    assert not shape.from_wkb_hex(wkb_hex, default_crs).y


def test_from_wkb_hex_raises():
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    wkb_hex = shapely.wkb.dumps(shapely.from_wkt(wkt), hex=True)
    default_crs = None
    with u.raises(Exception):
        shape.from_wkb_hex(wkb_hex, default_crs)


def test_from_ewkb_hex():
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    ewkb_hex = shapely.wkb.dumps(shapely.from_wkt(wkt), hex=True, srid=4326)
    assert shape.from_wkb_hex(ewkb_hex).geom == shapely.wkb.loads(bytes.fromhex(ewkb_hex))
    assert shape.from_wkb_hex(ewkb_hex).crs.epsg == 'EPSG:4326'
    assert shape.from_wkb_hex(ewkb_hex).type == 'polygon'
    assert not shape.from_wkb_hex(ewkb_hex).x
    assert not shape.from_wkb_hex(ewkb_hex).y


# def test_from_wkb_element():
#     wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
#     default_crs = crs.WGS84
#     wkb_hex = shapely.wkb.dumps(shapely.from_wkt(wkt), hex=True, srid=4326)
#     shp= shapely.wkb.loads(bytes.fromhex(wkb_hex))
#     element = geoalchemy2.shape.from_shape(shp)
#     assert shape.from_wkb_element(element, default_crs).geom == shp
#     assert not shape.from_wkb_element(element).crs.epsg
#     assert shape.from_wkb_element(element).type == 'polygon'
#     assert not shape.from_wkb_element(element).x
#     assert not shape.from_wkb_element(element).y

def test_from_geojson():
    geojson = {"type": "Polygon",
               "coordinates": [
                   [
                       [0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0],
                       [1.0, 0.0],
                       [0.0, 0.0]
                   ]
               ]}
    geojson_str = '''{"type": "Polygon",
  "coordinates": [
    [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0],
      [0.0, 0.0]
    ]
  ]}'''
    default_crs = crs.WGS84
    assert shape.from_geojson(geojson, default_crs).geom == shapely.from_geojson(geojson_str)
    assert shape.from_geojson(geojson, default_crs).crs.epsg == 'EPSG:4326'
    assert shape.from_geojson(geojson, default_crs).type == 'polygon'
    assert not shape.from_geojson(geojson, default_crs).x
    assert not shape.from_geojson(geojson, default_crs).y


def test_from_geojson_xy():
    geojson = {"type": "Polygon",
               "coordinates": [
                   [
                       [0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0],
                       [1.0, 0.0],
                       [0.0, 0.0]
                   ]
               ]}
    geojson_str = '''{"type": "Polygon",
  "coordinates": [
    [
      [0.0, 0.0],
      [0.0, 1.0],
      [1.0, 1.0],
      [1.0, 0.0],
      [0.0, 0.0]
    ]
  ]}'''
    default_crs = crs.WGS84
    assert shape.from_geojson(geojson, default_crs, always_xy=True).geom == shapely.from_geojson(geojson_str)
    assert shape.from_geojson(geojson, default_crs).crs.epsg == 'EPSG:4326'
    assert shape.from_geojson(geojson, default_crs).type == 'polygon'
    assert not shape.from_geojson(geojson, default_crs).x
    assert not shape.from_geojson(geojson, default_crs).y


def test_from_props():
    geojson = {"type": "Polygon",
               "coordinates": [
                   [
                       [0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0],
                       [1.0, 0.0],
                       [0.0, 0.0]
                   ]
               ]}
    geojson_str = '''{"type": "Polygon",
      "coordinates": [
        [
          [0.0, 0.0],
          [0.0, 1.0],
          [1.0, 1.0],
          [1.0, 0.0],
          [0.0, 0.0]
        ]
      ]}'''
    d = {'crs': '4326', 'geometry': geojson}
    data = gws.Data(d)
    props = gws.Props(data)
    assert shape.from_props(props).geom == shapely.from_geojson(geojson_str)
    assert shape.from_props(props).crs.epsg == 'EPSG:4326'
    assert shape.from_props(props).type == 'polygon'
    assert not shape.from_props(props).x
    assert not shape.from_props(props).y


def test_from_props_raises():
    d = {'crs': None}
    data = gws.Data(d)
    props = gws.Props(data)
    with u.raises(Exception):
        shape.from_props(props)


def test_from_dict():
    geojson = {"type": "Polygon",
               "coordinates": [
                   [
                       [0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0],
                       [1.0, 0.0],
                       [0.0, 0.0]
                   ]
               ]}
    geojson_str = '''{"type": "Polygon",
          "coordinates": [
            [
              [0.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0],
              [1.0, 0.0],
              [0.0, 0.0]
            ]
          ]}'''
    d = {'crs': '4326',
         'geometry': geojson}
    assert shape.from_dict(d).geom == shapely.from_geojson(geojson_str)
    assert shape.from_dict(d).crs.epsg == 'EPSG:4326'
    assert shape.from_dict(d).type == 'polygon'
    assert not shape.from_dict(d).x
    assert not shape.from_dict(d).y


def test_from_dict_raises():
    d = {'crs': None}
    with u.raises(Exception):
        shape.from_dict(d)


def test_from_from_bounds():
    geojson_str = '''{"type": "Polygon",
             "coordinates": [
               [
                 [1.0, 0.0],
                 [1.0, 1.0],
                 [0.0, 1.0],
                 [0.0, 0.0],
                 [1.0, 0.0]
               ]
             ]}'''
    c = crs.WGS84
    extent = gws.Extent((0.0, 0.0, 1.0, 1.0))
    bounds = gws.Bounds(crs=c, extent=extent)
    assert shape.from_bounds(bounds).geom == shapely.from_geojson(geojson_str)
    assert shape.from_bounds(bounds).crs.epsg == 'EPSG:4326'
    assert shape.from_bounds(bounds).type == 'polygon'
    assert not shape.from_bounds(bounds).x
    assert not shape.from_bounds(bounds).y


def test_from_xy():
    geojson_str = '''{"type": "Point",
                    "coordinates": [1.0, 1.0]}'''
    c = crs.WGS84
    assert shape.from_xy(1.0, 1.0, c).geom == shapely.from_geojson(geojson_str)
    assert shape.from_xy(1.0, 1.0, c).crs.epsg == 'EPSG:4326'
    assert shape.from_xy(1.0, 1.0, c).type == 'point'
    assert shape.from_xy(1.0, 1.0, c).x == 1.0
    assert shape.from_xy(1.0, 1.0, c).y == 1.0
