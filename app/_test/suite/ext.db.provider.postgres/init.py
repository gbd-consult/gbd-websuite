import _test.util as u
import _test.common.const as cc


def main():
    schema = {
        'p_str': 'text',
        'p_int': 'int',
        'p_date': 'date',
    }

    u.make_geom_table(
        name='points_3857',
        geom_type='point',
        prop_schema=schema,
        crs='EPSG:3857',
        rows=10,
        cols=5,
        start_x=cc.POI.tour_eiffel_3857_x,
        start_y=cc.POI.tour_eiffel_3857_y,
        gap=100,
    )

    u.make_geom_table(
        name='squares_3857',
        geom_type='square',
        prop_schema=schema,
        crs='EPSG:3857',
        rows=10,
        cols=5,
        start_x=cc.POI.flatiron_building_3857_x,
        start_y=cc.POI.flatiron_building_3857_y,
        gap=100,
    )

    u.make_geom_table(
        name='squares_25832',
        geom_type='square',
        prop_schema=schema,
        crs='EPSG:25832',
        rows=10,
        cols=5,
        start_x=cc.POI.big_ben_25832_x,
        start_y=cc.POI.big_ben_25832_y,
        gap=100,
    )


if __name__ == '__main__':
    main()
