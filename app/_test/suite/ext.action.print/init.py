import _test.util as u
import _test.common.const as cc


def main():
    schema = {
        'p_str': 'text',
        'p_int': 'int',
        'p_date': 'date',
    }

    u.make_geom_table(
        name='dus1',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_3857,
        rows=10,
        cols=5,
        xy=cc.POINTS.dus1,
        gap=100,
    )
    u.make_geom_table(
        name='dus2',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_3857,
        rows=10,
        cols=5,
        xy=cc.POINTS.dus2,
        gap=100,
    )
    u.make_geom_json(
        path='/common/qgis/points_dus3_3857.geojson',
        geom_type='point',
        prop_schema=schema,
        crs=cc.CRS_3857,
        rows=10,
        cols=5,
        xy=cc.POINTS.dus3,
        gap=100,
    )


if __name__ == '__main__':
    main()
