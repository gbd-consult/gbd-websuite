import _test.util as u
import _test.common.const as cc


def main():
    schema = {
        'p_str': 'text',
        'p_int': 'int',
        'p_date': 'date',
    }

    u.make_geom_json(
        path='/common/qgis/points_ghana_25832.geojson',
        geom_type='point',
        prop_schema=schema,
        crs=cc.CRS_25832,
        rows=10,
        cols=5,
        xy=cc.POINTS.ghana,
        gap=100,
    )
    u.make_geom_json(
        path='/common/qgis/squares_memphis_25832.geojson',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_25832,
        rows=10,
        cols=5,
        xy=cc.POINTS.memphis,
        gap=100,
    )
    u.make_geom_json(
        path='/common/qgis/squares_mexico_25832.geojson',
        geom_type='square',
        prop_schema=schema,
        crs=cc.CRS_25832,
        rows=70,
        cols=70,
        xy=cc.POINTS.mexico,
        gap=100,
    )


if __name__ == '__main__':
    main()
