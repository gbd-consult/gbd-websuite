"""Tests for GDAL vector data handling."""

import gws
import gws.test.util as u
import gws.gis.gdalx as gdalx
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
            c_date=datetimex.parse(f'2021-04-0{i}'),
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
            c_date=datetimex.parse(f'2021-04-0{i}'),
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
