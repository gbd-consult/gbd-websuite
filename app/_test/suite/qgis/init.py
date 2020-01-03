import _test.util as u
import _test.common.const as cc


def main():
    schema = {
        'p_str': 'text',
        'p_int': 'int',
        'p_date': 'date',
    }

    u.make_geom_table(
        name='squares_ny_2263',
        geom_type='square',
        prop_schema=schema,
        crs='EPSG:2263',
        rows=10,
        cols=5,
        xy=cc.POINTS.ny,
        gap=100,
    )
    u.make_geom_table(
        name='squares_dus1_3857',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_3857,
        rows=10,
        cols=5,
        xy=cc.POINTS.dus1,
        gap=100,
    )
    u.make_geom_table(
        name='squares_dus2_3857',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_3857,
        rows=10,
        cols=5,
        xy=cc.POINTS.dus2,
        gap=100,
    )


if __name__ == '__main__':
    main()
