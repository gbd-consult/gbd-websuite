import pytest

import gws
import gws.test.util as u
import gws.plugin.postgres.provider as postgres_provider


##
# Fixtures
##


@u.fixture(scope='module')
def root():
    u.pg.create('test_types', {
        'id': 'int primary key',
        'val_int': 'int',
        'val_text': 'text',
        'val_float': 'double precision',
        'val_bool': 'boolean',
        'val_date': 'date',
        'val_time': 'time',
        'val_ts': 'timestamp',
        'val_bytea': 'bytea',
        'val_intarr': 'int[]',
        'val_textarr': 'text[]',
    })
    u.pg.create('test_geom_point', {
        'id': 'int primary key',
        'geom': 'geometry(Point, 25832)',
    })
    u.pg.create('test_geom_poly', {
        'id': 'int primary key',
        'geom': 'geometry(Polygon, 25832)',
    })
    u.pg.create('test_no_geom', {
        'id': 'int primary key',
        'name': 'text',
    })

    yield u.gws_root()


@u.fixture(autouse=True)
def clear_tables():
    yield
    u.pg.clear('test_types')
    u.pg.clear('test_geom_point')
    u.pg.clear('test_geom_poly')
    u.pg.clear('test_no_geom')


##
# split_table_name
##


def test_split_simple_name():
    prov = postgres_provider.Object.__new__(postgres_provider.Object)
    schema, name = prov.split_table_name('mytable')
    assert schema == 'public'
    assert name == 'mytable'


def test_split_schema_and_name():
    prov = postgres_provider.Object.__new__(postgres_provider.Object)
    schema, name = prov.split_table_name('myschema.mytable')
    assert schema == 'myschema'
    assert name == 'mytable'


def test_split_quoted_name():
    prov = postgres_provider.Object.__new__(postgres_provider.Object)
    schema, name = prov.split_table_name('"My Schema"."My Table"')
    assert schema == 'My Schema'
    assert name == 'My Table'


def test_split_quoted_name_with_escaped_quotes():
    prov = postgres_provider.Object.__new__(postgres_provider.Object)
    schema, name = prov.split_table_name('"schema""x"."table""y"')
    assert schema == 'schema"x'
    assert name == 'table"y'


def test_split_mixed_quoted():
    prov = postgres_provider.Object.__new__(postgres_provider.Object)
    schema, name = prov.split_table_name('myschema."My Table"')
    assert schema == 'myschema'
    assert name == 'My Table'


def test_split_invalid_name():
    prov = postgres_provider.Object.__new__(postgres_provider.Object)
    with u.raises(ValueError):
        prov.split_table_name('')


##
# join_table_name
##


def test_join_with_schema():
    prov = postgres_provider.Object.__new__(postgres_provider.Object)
    result = prov.join_table_name('myschema', 'mytable')
    assert result == 'myschema.mytable'


def test_join_empty_schema_uses_split():
    prov = postgres_provider.Object.__new__(postgres_provider.Object)
    result = prov.join_table_name('', 'public.mytable')
    assert result == 'public.mytable'


##
# connection_url
##


def test_connection_url_with_host():
    cfg = gws.Config(
        host='localhost',
        port=5432,
        username='user1',
        password='pass1',
        database='mydb',
        options=None,
    )
    url = postgres_provider.connection_url(cfg)
    assert url is not None
    assert 'postgresql' in url
    assert 'localhost' in url
    assert 'mydb' in url
    assert 'user1' in url


def test_connection_url_no_host_no_service_returns_empty():
    cfg = gws.Config(
        host=None,
        port=5432,
        username='user1',
        password='pass1',
        database='mydb',
        serviceName=None,
        options=None,
    )
    url = postgres_provider.connection_url(cfg)
    assert url is None


##
# Provider integration via gws_root
##


def test_has_table_true(root: gws.Root):
    db = u.get_db(root)
    assert db.has_table('test_types') is True


def test_has_table_false(root: gws.Root):
    db = u.get_db(root)
    assert db.has_table('no_such_table_xyz') is False


def test_has_column_true(root: gws.Root):
    db = u.get_db(root)
    assert db.has_column('test_types', 'val_int') is True


def test_has_column_false(root: gws.Root):
    db = u.get_db(root)
    assert db.has_column('test_types', 'no_such_col') is False


##
# describe_column – basic types
##


def test_describe_column_int(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_int')
    assert col.type == gws.AttributeType.int


def test_describe_column_text(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_text')
    assert col.type == gws.AttributeType.str


def test_describe_column_float(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_float')
    assert col.type == gws.AttributeType.float


def test_describe_column_bool(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_bool')
    assert col.type == gws.AttributeType.bool


def test_describe_column_date(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_date')
    assert col.type == gws.AttributeType.date


def test_describe_column_time(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_time')
    assert col.type == gws.AttributeType.time


def test_describe_column_timestamp(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_ts')
    assert col.type == gws.AttributeType.datetime


def test_describe_column_bytea(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_bytea')
    assert col.type == gws.AttributeType.bytes


def test_describe_column_primary_key(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'id')
    assert col.isPrimaryKey is True


def test_describe_column_non_pk(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_int')
    assert col.isPrimaryKey is False


##
# describe_column – arrays
##


def test_describe_column_int_array(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_intarr')
    assert col.type == gws.AttributeType.intlist


def test_describe_column_text_array(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_types', 'val_textarr')
    assert col.type == gws.AttributeType.strlist


##
# describe_column – geometry
##


def test_describe_column_geometry(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_geom_point', 'geom')
    assert col.type == gws.AttributeType.geometry
    assert col.geometryType == gws.GeometryType.point
    assert col.geometrySrid == 25832


def test_describe_column_geometry_polygon(root: gws.Root):
    db = u.get_db(root)
    col = db.describe_column('test_geom_poly', 'geom')
    assert col.type == gws.AttributeType.geometry
    assert col.geometryType == gws.GeometryType.polygon
    assert col.geometrySrid == 25832


##
# describe – table-level
##


def test_describe_no_geometry(root: gws.Root):
    db = u.get_db(root)
    desc = db.describe('test_no_geom')
    assert desc.geometryName == ''
    assert len(desc.columns) == 2
    assert 'id' in desc.columnMap
    assert 'name' in desc.columnMap


def test_describe_with_geometry(root: gws.Root):
    db = u.get_db(root)
    desc = db.describe('test_geom_point')
    assert desc.geometryName == 'geom'
    assert desc.geometryType == gws.GeometryType.point
    assert desc.geometrySrid == 25832


def test_describe_full_name(root: gws.Root):
    db = u.get_db(root)
    desc = db.describe('test_no_geom')
    assert desc.fullName == 'public.test_no_geom'


##
# table_bounds
##


def test_table_bounds_no_geometry(root: gws.Root):
    db = u.get_db(root)
    result = db.table_bounds('test_no_geom')
    assert result is None


def test_table_bounds_empty_table(root: gws.Root):
    db = u.get_db(root)
    # table exists and has geometry column but is empty
    result = db.table_bounds('test_geom_point')
    assert result is None


def test_table_bounds_with_data(root: gws.Root):
    # Use a polygon so ST_Extent returns a non-degenerate box (min != max).
    # A single point yields BOX(x y,x y) where min==max, which from_box rejects.
    u.pg.exec(
        "INSERT INTO test_geom_poly(id, geom) VALUES(1, "
        "ST_GeomFromText('POLYGON((400000 5700000, 401000 5700000, 401000 5701000, 400000 5700000))', 25832))"
    )
    db = u.get_db(root)
    result = db.table_bounds('test_geom_poly')
    assert result is not None
    assert result.crs is not None
    assert result.extent is not None


def test_table_bounds_single_point_is_none(root: gws.Root):
    # A single point yields a degenerate box where min==max.
    # from_box returns None in that case, so table_bounds returns None too.
    u.pg.exec(
        "INSERT INTO test_geom_point(id, geom) VALUES(1, ST_GeomFromText('POINT(400000 5700000)', 25832))"
    )
    db = u.get_db(root)
    result = db.table_bounds('test_geom_point')
    assert result is None


##
# schema_names / has_schema
##


def test_has_schema_public(root: gws.Root):
    db = u.get_db(root)
    assert db.has_schema('public') is True


def test_has_schema_nonexistent(root: gws.Root):
    db = u.get_db(root)
    assert db.has_schema('no_such_schema_xyz') is False
