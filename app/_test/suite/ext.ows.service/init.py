import _test.util as u
import _test.common.const as cc


def main():
    schema = {
        'p_str': 'text',
        'p_int': 'int',
        'p_date': 'date',
    }

    u.make_geom_table(
        name='paris_3857',
        geom_type='point',
        prop_schema=schema,
        crs=cc.CRS_3857,
        rows=10,
        cols=5,
        xy=cc.POINTS.paris,
        gap=100,
    )

    u.make_geom_table(
        name='dus_25832',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_25832,
        rows=10,
        cols=5,
        xy=cc.POINTS.dus,
        gap=100,
    )

    u.make_geom_table(
        name='ny_3857',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_3857,
        rows=10,
        cols=5,
        xy=cc.POINTS.ny,
        gap=100,
    )

    u.make_geom_table(
        name='london_25833',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_25833,
        rows=10,
        cols=5,
        xy=cc.POINTS.london,
        gap=100,
    )


if __name__ == '__main__':
    main()
