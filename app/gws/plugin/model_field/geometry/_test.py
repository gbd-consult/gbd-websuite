import gws.base.shape
import gws.test.util as u
import gws.lib.crs


@u.fixture(scope='module')
def model():
    u.pg.create('geometry_table', {'id': 'int primary key', 'geom': 'geometry(Point, 4326)'})

    cfg = """
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
        """

    root = u.gws_root(cfg)
    yield u.cast(gws.Model, root.get('GEOMETRY_MODEL'))


def test_create(model: gws.Model):
    mc = u.model_context()

    point1 = gws.base.shape.from_xy(11, 22, gws.lib.crs.WGS84)
    point2 = gws.base.shape.from_xy(33, 44, gws.lib.crs.WGS84)

    f = u.feature(model, id=1, geom=point1)
    model.create_feature(f, mc)

    f = u.feature(model, id=2, geom=point2)
    model.create_feature(f, mc)

    features = model.get_features([1, 2], mc)

    shape = gws.u.require(features[0].shape())
    assert shape.to_ewkt(trim=True) == 'SRID=4326;POINT(11 22)'

    shape = gws.u.require(features[1].shape())
    assert shape.to_ewkt(trim=True) == 'SRID=4326;POINT(33 44)'


def test_read(model: gws.Model):
    mc = u.model_context()

    point1 = gws.base.shape.from_xy(123, 456, gws.lib.crs.WGS84)
    u.pg.insert('geometry_table', [{'id': 1, 'geom': point1.to_ewkb_hex()}])

    features = model.get_features([1], mc)
    assert len(features) == 1

    shape = gws.u.require(features[0].shape())
    assert shape.to_ewkt(trim=True) == 'SRID=4326;POINT(123 456)'


def test_update(model: gws.Model):
    mc = u.model_context()

    point1 = gws.base.shape.from_xy(11, 22, gws.lib.crs.WGS84)
    point2 = gws.base.shape.from_xy(77, 99, gws.lib.crs.WGS84)

    f = u.feature(model, id=99, geom=point1)
    model.create_feature(f, mc)

    f = u.feature(model, id=99, geom=point2)
    model.update_feature(f, mc)

    features = model.get_features([99], mc)

    shape = gws.u.require(features[0].shape())
    assert shape.to_ewkt(trim=True) == 'SRID=4326;POINT(77 99)'
