from pyexpat import features

import gws.config
import gws.base.feature
import gws.base.shape
import gws.test.util as u
import gws.gis.crs


@u.fixture(scope='module')
def root():
    u.pg.create('geometry_table', {'id': 'int primary key', 'geom': 'geometry(Point, 4326)'})

    cfg = '''
            models+ {
                uid "GEOMETRY_MODEL" type "postgres" tableName "geometry_table"
                fields+ { name "id" type "integer" }
                fields+ { 
                    name "geom" 
                    type geometry 
                    geometryType "Point" 
                    crs "EPSG:4326" 
                }
            }
        '''

    yield u.gws_root(cfg)


def test_create_geometry(root: gws.Root):
    mc = u.model_context()

    model = u.cast(gws.Model, root.get('GEOMETRY_MODEL'))

    point = gws.Shape()
    point.type = gws.GeometryType.point
    point.crs = gws.gis.crs.WGS84
    point.x = 1
    point.y = 2

    point2 = gws.Shape()
    point2.type = gws.GeometryType.point
    point2.crs = gws.gis.crs.WGS84
    point2.x = 2
    point2.y = 3

    f = u.feature(model, id=1, geom=point)
    model.create_feature(f, mc)

    f = u.feature(model, id=2, geom=point2)
    model.create_feature(f, mc)

    features = model.get_features([1,2], mc)

    geom = features[0].get('geom')

    shape = gws.base.shape.from_wkb_hex(geom.to_ewkb_hex())
    assert shape.to_props().get('crs') == 'EPSG:4326'
    assert shape.to_props().get('geometry').get('type') == 'Point'
    assert shape.to_props().get('geometry').get('coordinates') == (1.0, 2.0)


    geom = features[1].get('geom')

    shape = gws.base.shape.from_wkb_hex(geom.to_ewkb_hex())
    assert shape.to_props().get('crs') == 'EPSG:4326'
    assert shape.to_props().get('geometry').get('type') == 'Point'
    assert shape.to_props().get('geometry').get('coordinates') == (2.0, 3.0)


def test_read_geometry(root: gws.Root):
    mc = u.model_context()

    model = u.cast(gws.Model, root.get('GEOMETRY_MODEL'))

    point_wkb = '0101000000000000000000F03F0000000000000040'  # WKB for POINT(1 2)
    u.pg.insert('geometry_table', [{'id': 1, 'geom': point_wkb}])

    features = model.get_features([1], mc)
    assert len(features) == 1

    geom = features[0].get('geom')
    shape = gws.base.shape.from_wkb_hex(geom.to_ewkb_hex())
    assert shape.to_props().get('crs') == 'EPSG:4326'
    assert shape.to_props().get('geometry').get('type') == 'Point'
    assert shape.to_props().get('geometry').get('coordinates') == (1.0, 2.0)


def test_update_geometry(root: gws.Root):
    mc = u.model_context()

    model = u.cast(gws.Model, root.get('GEOMETRY_MODEL'))

    point_wkb = '0101000000000000000000F03F0000000000000040'  # WKB for POINT(1 2)
    point2_wkb = '0101000000000000000000F03F0000000000000040'  # WKB for POINT(1 2)
    u.pg.insert('geometry_table', [{'id': 1, 'geom': point_wkb}, {'id': 2, 'geom': point2_wkb}])

    point = gws.Shape()
    point.type = gws.GeometryType.point
    point.crs = gws.gis.crs.WGS84
    point.x = 3
    point.y = 4

    f = u.feature(model, id=2, geom=point)
    model.update_feature(f, mc)

    rows = u.pg.rows('SELECT id, geom FROM geometry_table ORDER BY id')
    point_ewkb = '0101000020E6100000000000000000F03F0000000000000040'
    point2_ewkb = '0101000020E610000000000000000008400000000000001040'
    assert rows == [
        (1, point_ewkb), (2, point2_ewkb)
    ]


def test_delete_geometry(root: gws.Root):
    mc = u.model_context()

    point_wkb = '0101000000000000000000F03F0000000000000040'  # WKB for POINT(1 2)
    point2_wkb = '0101000000000000000000F03F0000000000000040'  # WKB for POINT(1 2)
    u.pg.insert('geometry_table', [{'id': 1, 'geom': point_wkb}, {'id': 2, 'geom': point2_wkb}])

    model = u.cast(gws.Model, root.get('GEOMETRY_MODEL'))

    f = u.feature(model, id=1)
    model.delete_feature(f, mc)

    rows = u.pg.rows('SELECT id, geom FROM geometry_table ORDER BY id')
    point2_ewkb = '0101000020E6100000000000000000F03F0000000000000040'
    assert rows == [(2, point2_ewkb)]

    f = u.feature(model, id=2)
    model.delete_feature(f, mc)

    rows = u.pg.rows('SELECT id, geom FROM geometry_table ORDER BY id')
    assert rows == []