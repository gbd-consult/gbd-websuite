import gws
import gws.lib.image
import gws.test.util as u
import gws.lib.crs
import gws.gis.ms as ms


@u.fixture(scope='module')
def db():
    root = u.gws_root()
    yield u.get_db(root)


def test_postgres_lines(db: gws.DatabaseProvider):
    tab = 'lines'

    u.pg.create(tab, {'id': 'int primary key', 'geom': 'geometry(LINESTRING, 4326)'})
    u.pg.insert(
        tab,
        [
            {'id': 1, 'geom': 'LINESTRING(1 3, 6 7)'},
            {'id': 2, 'geom': 'LINESTRING(7 6, 3 1)'},
        ],
    )

    cfg = f"""
        LAYER
            STATUS ON
            TYPE LINE
            CONNECTIONTYPE POSTGIS
            CONNECTION "{db.url()}"
            DATA "geom FROM {tab} USING UNIQUE id USING SRID=4326"
            CLASS
                STYLE
                    COLOR 255 255 0
                    WIDTH 2
                END
            END
        END
    """

    mm = ms.new_map()
    mm.add_layer(cfg)
    img = mm.draw(
        gws.Bounds(extent=(1, 1, 10, 10), crs=gws.lib.crs.WGS84),
        (200, 200),
    )
    # 
    # img.to_path('/data/test_create.png')
    assert img.compare_to(gws.lib.image.from_path('/data/test_create.png')) > 0.99 
    
