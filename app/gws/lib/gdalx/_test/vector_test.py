"""Tests for GDAL vector data handling."""

import datetime
import gws
import gws.test.util as u
import gws.lib.gdalx as gdalx
import gws.base.shape
import gws.lib.datetimex as datetimex
import gws.lib.crs
import gws.lib.osx as osx


def test_shp():
    cols = dict(
        c_date=gws.AttributeType.date,
        c_float=gws.AttributeType.float,
        c_int=gws.AttributeType.int,
        c_str=gws.AttributeType.str,
    )
    crs = gws.lib.crs.require(25833)

    recs_a = []

    for i in range(1, 10):
        rec = gws.FeatureRecord()
        rec.attributes = dict(
            c_date=datetime.date(2021, 4, i),
            c_float=i / 10,
            c_int=i,
            c_str=f'~D|∂|ü|Ю~{i}',
        )
        rec.shape = gws.base.shape.from_xy(i * 1000, i * 2000, crs)
        recs_a.append(rec)

    with u.temp_dir_in_base_dir() as d:
        with gdalx.open_vector(f'{d}/shape.shp', 'w') as ds:
            la = ds.create_layer('', cols, gws.GeometryType.point, crs)
            la.insert(recs_a)
        dbf = open(f'{d}/shape.dbf', 'rb').read()
        assert recs_a[0].attributes['c_str'].encode('utf8') in dbf

        with gdalx.open_vector(f'{d}/shape.shp', 'r') as ds:
            la = ds.layer(0)
            assert la is not None
            recs_b = la.get_all()

        assert [r.attributes for r in recs_a] == [r.attributes for r in recs_b]
        assert [r.shape.to_ewkt() for r in recs_a] == [r.shape.to_ewkt() for r in recs_b]


def test_shp_with_encoding():
    cols = dict(
        name=gws.AttributeType.str,
    )

    recs_a = []

    names = [f'~D|ä|ü|ß~']

    for name in names:
        rec = gws.FeatureRecord()
        rec.attributes = dict(name=name)
        recs_a.append(rec)

    with u.temp_dir_in_base_dir() as d:
        with gdalx.open_vector(f'{d}/shape_iso.shp', 'w', encoding='ISO-8859-1') as ds:
            la = ds.create_layer('', cols)
            la.insert(recs_a)

        dbf = open(f'{d}/shape_iso.dbf', 'rb').read()
        assert names[0].encode('ISO-8859-1') in dbf

        # NB: "encoding" should only be passed when no "cpg" file is present
        osx.unlink(f'{d}/shape_iso.cpg')

        with gdalx.open_vector(f'{d}/shape_iso.shp', 'r', encoding='ISO-8859-1') as ds:
            la = ds.layer(0)
            assert [r.attributes for r in recs_a] == [r.attributes for r in la.get_all()]

        with gdalx.open_vector(f'{d}/shape_utf.shp', 'w', encoding='utf8') as ds:
            la = ds.create_layer('', cols)
            la.insert(recs_a)

        dbf = open(f'{d}/shape_utf.dbf', 'rb').read()
        assert names[0].encode('utf8') in dbf

        # NB: "encoding" should only be passed when no "cpg" file is present
        osx.unlink(f'{d}/shape_utf.cpg')

        with gdalx.open_vector(f'{d}/shape_utf.shp', 'r', encoding='UTF8') as ds:
            la = ds.layer(0)
            assert [r.attributes for r in recs_a] == [r.attributes for r in la.get_all()]


def test_gpkg():
    cols = dict(
        c_date=gws.AttributeType.date,
        c_float=gws.AttributeType.float,
        c_int=gws.AttributeType.int,
        c_str=gws.AttributeType.str,
    )
    crs = gws.lib.crs.require(25833)

    recs_a = []

    for i in range(1, 10):
        rec = gws.FeatureRecord()
        rec.attributes = dict(
            c_date=datetime.date(2021, 4, i),
            c_float=i / 10,
            c_int=i,
            c_str=f'~D|∂|ü|Ю~{i}',
        )
        rec.shape = gws.base.shape.from_xy(i * 1000, i * 2000, crs)
        recs_a.append(rec)

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/data.gpkg'
        with gdalx.open_vector(path, 'w') as ds:
            la = ds.create_layer('', cols, gws.GeometryType.point, crs)
            la.insert(recs_a)

        with gdalx.open_vector(path, 'r') as ds:
            la = ds.layer(0)
            assert la is not None
            recs_b = la.get_all()

        assert [r.attributes for r in recs_a] == [r.attributes for r in recs_b]
        assert [r.shape.to_ewkt() for r in recs_a] == [r.shape.to_ewkt() for r in recs_b]


def test_gpkg_bool():
    crs = gws.lib.crs.require(25833)
    cols = dict(c_bool=gws.AttributeType.bool)

    recs_a = [
        gws.FeatureRecord(attributes=dict(c_bool=True)),
        gws.FeatureRecord(attributes=dict(c_bool=False)),
        gws.FeatureRecord(attributes=dict(c_bool=None)),   # null value
        gws.FeatureRecord(attributes=dict(c_bool=1)),      # integer truthy input
        gws.FeatureRecord(attributes=dict(c_bool=0)),      # integer falsy input
    ]

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/bool.gpkg'
        with gdalx.open_vector(path, 'w') as ds:
            la = ds.create_layer('test', cols, gws.GeometryType.point, crs)
            la.insert(recs_a)

        with gdalx.open_vector(path, 'r') as ds:
            la = ds.layer(0)
            desc = la.describe()
            assert desc.columnMap['c_bool'].type == gws.AttributeType.bool
            recs_b = la.get_all()

    # True/False must come back as Python bool, not int
    assert recs_b[0].attributes['c_bool'] is True
    assert type(recs_b[0].attributes['c_bool']) is bool
    assert recs_b[1].attributes['c_bool'] is False
    assert type(recs_b[1].attributes['c_bool']) is bool
    # null -> None
    assert recs_b[2].attributes['c_bool'] is None
    # integer 1/0 written to a bool field must also read back as bool
    assert recs_b[3].attributes['c_bool'] is True
    assert recs_b[4].attributes['c_bool'] is False


def test_gpkg_bytes():
    crs = gws.lib.crs.require(25833)
    cols = dict(c_bytes=gws.AttributeType.bytes)

    data = b'\x00\x01\x02\xfe\xff'
    recs_a = [gws.FeatureRecord(attributes=dict(c_bytes=data))]

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/bytes.gpkg'
        with gdalx.open_vector(path, 'w') as ds:
            la = ds.create_layer('test', cols, gws.GeometryType.point, crs)
            la.insert(recs_a)

        with gdalx.open_vector(path, 'r') as ds:
            recs_b = ds.layer(0).get_all()

    assert recs_b[0].attributes['c_bytes'] == data


def test_gpkg_datetime():
    crs = gws.lib.crs.require(25833)
    cols = dict(c_dt=gws.AttributeType.datetime)

    dt_val = datetimex.parse('2024-06-15T12:30:45')
    recs_a = [gws.FeatureRecord(attributes=dict(c_dt=dt_val))]

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/datetime.gpkg'
        with gdalx.open_vector(path, 'w') as ds:
            la = ds.create_layer('test', cols, gws.GeometryType.point, crs)
            la.insert(recs_a)

        with gdalx.open_vector(path, 'r') as ds:
            recs_b = ds.layer(0).get_all()

    result = recs_b[0].attributes['c_dt']
    assert isinstance(result, datetime.datetime)
    assert result == dt_val


def test_geojson_time():
    # GPKG stores OFTTime as a TEXT column (OFTString in OGR), so the value
    # comes back as an ISO time string rather than a dt.time object.
    # This test verifies the written value round-trips correctly at the string level.
    crs = gws.lib.crs.require(25833)
    cols = dict(c_time=gws.AttributeType.time)

    time_val = datetime.time(14, 30, 45)
    rec_a = gws.FeatureRecord(attributes=dict(c_time=time_val))
    rec_a.shape = gws.base.shape.from_xy(1000, 2000, crs)

    with u.temp_dir_in_base_dir() as d:
        with gdalx.open_vector(f'{d}/time.gpkg', 'w') as ds:
            la = ds.create_layer('test', cols, gws.GeometryType.point, crs)
            la.insert([rec_a])

        with gdalx.open_vector(f'{d}/time.gpkg', 'r') as ds:
            recs_b = ds.layer(0).get_all()

    result = recs_b[0].attributes['c_time']
    assert result is not None
    # GPKG returns the value as OFTString; parse it back to check h/m/s
    if isinstance(result, datetime.time):
        parsed = result
    elif isinstance(result, datetime.datetime):
        parsed = result.time()
    else:
        parsed = datetimex.parse_time(str(result))
    assert parsed.hour == 14
    assert parsed.minute == 30
    assert parsed.second == 45


def test_geojson_floatlist():
    # GeoJSON natively supports JSON number arrays → OFTRealList round-trips correctly.
    crs = gws.lib.crs.require(25833)
    cols = dict(c_floatlist=gws.AttributeType.floatlist)

    vals = [1.1, 2.2, 3.3]
    rec = gws.FeatureRecord(attributes=dict(c_floatlist=vals))
    rec.shape = gws.base.shape.from_xy(1000, 2000, crs)

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/floatlist.geojson'
        with gdalx.open_vector(path, 'w') as ds:
            la = ds.create_layer('test', cols, gws.GeometryType.point, crs)
            la.insert([rec])

        with gdalx.open_vector(path, 'r') as ds:
            recs_b = ds.layer(0).get_all()

    result = recs_b[0].attributes['c_floatlist']
    assert len(result) == len(vals)
    for a, b in zip(result, vals):
        assert abs(a - b) < 1e-9


def test_geojson_intlist():
    # GeoJSON natively supports JSON integer arrays → OFTIntegerList round-trips correctly.
    crs = gws.lib.crs.require(25833)
    cols = dict(c_intlist=gws.AttributeType.intlist)

    vals = [10, 20, 30]
    rec = gws.FeatureRecord(attributes=dict(c_intlist=vals))
    rec.shape = gws.base.shape.from_xy(1000, 2000, crs)

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/intlist.geojson'
        with gdalx.open_vector(path, 'w') as ds:
            la = ds.create_layer('test', cols, gws.GeometryType.point, crs)
            la.insert([rec])

        with gdalx.open_vector(path, 'r') as ds:
            recs_b = ds.layer(0).get_all()

    assert recs_b[0].attributes['c_intlist'] == vals


def test_geojson_strlist():
    # GeoJSON natively supports JSON string arrays → OFTStringList round-trips correctly.
    crs = gws.lib.crs.require(25833)
    cols = dict(c_strlist=gws.AttributeType.strlist)

    vals = ['hello', 'world', 'unicode']
    rec = gws.FeatureRecord(attributes=dict(c_strlist=vals))
    rec.shape = gws.base.shape.from_xy(1000, 2000, crs)

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/strlist.geojson'
        with gdalx.open_vector(path, 'w') as ds:
            la = ds.create_layer('test', cols, gws.GeometryType.point, crs)
            la.insert([rec])

        with gdalx.open_vector(path, 'r') as ds:
            recs_b = ds.layer(0).get_all()

    assert recs_b[0].attributes['c_strlist'] == vals


def test_gpkg_multiple_layers():
    crs = gws.lib.crs.require(25833)
    cols = dict(name=gws.AttributeType.str)

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/multi.gpkg'
        with gdalx.open_vector(path, 'w') as ds:
            la1 = ds.create_layer('layer_a', cols, gws.GeometryType.point, crs)
            la1.insert([gws.FeatureRecord(attributes=dict(name='a1')),
                        gws.FeatureRecord(attributes=dict(name='a2'))])
            la2 = ds.create_layer('layer_b', cols, gws.GeometryType.point, crs)
            la2.insert([gws.FeatureRecord(attributes=dict(name='b1'))])

        with gdalx.open_vector(path, 'r') as ds:
            assert len(ds.layers()) == 2
            la = ds.layer('layer_a')
            assert la is not None
            assert la.count() == 2
            names_a = [r.attributes['name'] for r in la.get_all()]
            assert sorted(names_a) == ['a1', 'a2']
            la = ds.layer('layer_b')
            assert la.count() == 1


def test_gpkg_geometry_types():
    crs = gws.lib.crs.require(25833)
    cols = dict(name=gws.AttributeType.str)

    point_wkt = 'POINT (1000 2000)'
    line_wkt = 'LINESTRING (0 0, 1000 1000, 2000 0)'
    poly_wkt = 'POLYGON ((0 0, 1000 0, 1000 1000, 0 1000, 0 0))'

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/geomtypes.gpkg'
        with gdalx.open_vector(path, 'w') as ds:
            for name, wkt, geom_type in [
                ('pt', point_wkt, gws.GeometryType.point),
                ('ln', line_wkt, gws.GeometryType.linestring),
                ('pg', poly_wkt, gws.GeometryType.polygon),
            ]:
                la = ds.create_layer(name, cols, geom_type, crs)
                shape = gws.base.shape.from_wkt(wkt, crs)
                la.insert([gws.FeatureRecord(attributes=dict(name=name), shape=shape)])

        with gdalx.open_vector(path, 'r') as ds:
            pt_recs = ds.layer('pt').get_all()
            ln_recs = ds.layer('ln').get_all()
            pg_recs = ds.layer('pg').get_all()

    assert pt_recs[0].shape.type == gws.GeometryType.point
    assert ln_recs[0].shape.type == gws.GeometryType.linestring
    assert pg_recs[0].shape.type == gws.GeometryType.polygon


def test_gpkg_get_by_fid():
    crs = gws.lib.crs.require(25833)
    cols = dict(name=gws.AttributeType.str)

    recs_a = [
        gws.FeatureRecord(attributes=dict(name='first')),
        gws.FeatureRecord(attributes=dict(name='second')),
        gws.FeatureRecord(attributes=dict(name='third')),
    ]

    with u.temp_dir_in_base_dir() as d:
        path = f'{d}/fid.gpkg'
        with gdalx.open_vector(path, 'w') as ds:
            la = ds.create_layer('test', cols, gws.GeometryType.point, crs)
            fids = la.insert(recs_a)

        with gdalx.open_vector(path, 'r') as ds:
            la = ds.layer(0)
            assert la.count() == 3
            rec = la.get(fids[1])

    assert rec is not None
    assert rec.attributes['name'] == 'second'


def test_shp_bool():
    # SHP does not reliably preserve OFSTBoolean subtype across GDAL versions;
    # the field is stored as OFTInteger. Values must at least round-trip as 1/0.
    cols = dict(c_bool=gws.AttributeType.bool)
    crs = gws.lib.crs.require(25833)

    recs_a = [
        gws.FeatureRecord(attributes=dict(c_bool=True)),
        gws.FeatureRecord(attributes=dict(c_bool=False)),
        gws.FeatureRecord(attributes=dict(c_bool=1)),   # integer truthy input
        gws.FeatureRecord(attributes=dict(c_bool=0)),   # integer falsy input
    ]
    for rec in recs_a:
        rec.shape = gws.base.shape.from_xy(0, 0, crs)

    with u.temp_dir_in_base_dir() as d:
        with gdalx.open_vector(f'{d}/bool.shp', 'w') as ds:
            la = ds.create_layer('', cols, gws.GeometryType.point, crs)
            la.insert(recs_a)

        with gdalx.open_vector(f'{d}/bool.shp', 'r') as ds:
            la = ds.layer(0)
            desc = la.describe()
            # SHP may return bool or int depending on GDAL version
            assert desc.columnMap['c_bool'].type in (gws.AttributeType.bool, gws.AttributeType.int)
            recs_b = la.get_all()

    # Values must be truthy/falsy; type is bool if subtype preserved, else int
    assert bool(recs_b[0].attributes['c_bool']) is True
    assert bool(recs_b[1].attributes['c_bool']) is False
    assert bool(recs_b[2].attributes['c_bool']) is True
    assert bool(recs_b[3].attributes['c_bool']) is False
