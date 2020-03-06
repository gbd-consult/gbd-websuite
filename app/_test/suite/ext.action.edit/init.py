import _test.util as u
import _test.common.const as cc


def main():
    schema = {
        'p_str1': 'text',
        'p_str2': 'text',
        'p_int1': 'int',
        'p_int2': 'int',
        'p_date': 'date',
    }

    u.make_features(
        'postgres:paris_3857',
        geom_type='point',
        prop_schema=schema,
        crs=cc.CRS_3857,
        rows=10,
        cols=5,
        xy=cc.POINTS.paris,
        gap=100,
    )


if __name__ == '__main__':
    main()
