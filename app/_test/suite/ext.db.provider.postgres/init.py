import gws.tools.json2
import gws.ext.db.provider.postgres.provider

import gws.types as t

import _test.util as u


def main():
    cfg = u.test_config()['postgres']

    prov = gws.ext.db.provider.postgres.provider.Object()
    prov.initialize(t.Data(cfg))

    with prov.connect() as drv:
        drv.execute_many(
            ['drop table if exists points_3857'],
            ['''
                create table points_3857 (
                    id integer primary key,
                    p_str text,
                    p_int int,
                    p_date date,
                    p_geom geometry(point,3857)
                )
            ''']
        )

    schema = {
        'p_str': 'str',
        'p_int': 'int',
        'p_date': 'date',
    }

    crs = 'EPSG:3857'

    fs = u.make_point_features(
        schema=schema,
        crs=crs,
        rows=10,
        cols=5,
        start_x=1000, start_y=2000, gap=100,
    )

    prov.edit_operation('insert', t.SqlTable({
        'name': 'points_3857',
        'key_column': 'id',
        'geometry_column': 'p_geom',
        'geometry_crs': crs,
    }), fs)


if __name__ == '__main__':
    main()
