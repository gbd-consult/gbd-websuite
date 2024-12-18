"""gdalx tests."""

import os

import gws
import gws.test.util as u
import gws.gis.gdalx as gdalx
import gws.base.shape
import gws.lib.datetimex as datetimex
import gws.lib.crs


def test_shp():
    cols = dict(
        c_date=gws.AttributeType.date,
        c_float=gws.AttributeType.float,
        c_int=gws.AttributeType.int,
        c_str=gws.AttributeType.str,
    )

    recs_a = []

    for i in range(1, 10):
        rec = gws.FeatureRecord()
        rec.attributes = dict(
            c_date=datetimex.parse(f'2021-04-0{i}'),
            c_float=i / 10,
            c_int=i,
            c_str=f'{i}-{i}-{i}',
        )
        rec.shape = gws.base.shape.from_xy(i * 1000, i * 2000, gws.lib.crs.get(25833))
        recs_a.append(rec)

    with u.temp_dir_in_base_dir() as d:
        with gdalx.open_vector(f'{d}/shape.shp', 'w') as ds:
            la = ds.create_layer(
                '',
                cols,
                gws.GeometryType.point,
                gws.lib.crs.get(25833)
            )
            la.insert(recs_a)

        with gdalx.open_vector(f'{d}/shape.shp', 'r') as ds:
            la = ds.layer(0)
            recs_b = la.get_all()

        assert [r.attributes for r in recs_a] == [r.attributes for r in recs_b]
        assert [r.shape.to_ewkt() for r in recs_a] == [r.shape.to_ewkt() for r in recs_b]
