"""Tests the writer module by comparing its output to postgis ST_AsGML."""

import re

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.sa as sa
import gws.lib.gml
import gws.test.util as u

DATA = '''
POINT (30 10)
--
LINESTRING (30 10, 10 30, 40 40)
--
POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))
--
POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))
--
MULTIPOINT ((10 40), (40 30), (20 20), (30 10))
--
MULTIPOINT (10 40, 40 30, 20 20, 30 10)
--
MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))
--
MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))
--
MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (30 20, 20 15, 20 25, 30 20)))
--
GEOMETRYCOLLECTION (POINT (40 10), LINESTRING (10 10, 20 20, 10 40), POLYGON ((40 40, 20 45, 45 30, 40 40)))
'''


@u.fixture(scope='module')
def root():
    yield u.gws_root()


##

def test_with_postgis(root: gws.Root):
    db = u.get_db(root)
    wkt_list = []

    for s in DATA.strip().split('--'):
        wkt_list.append(re.sub(r'\s+', ' ', s.strip()))

    with db.connect() as conn:
        for version in [2, 3]:
            for wkt in wkt_list:
                sql = f"SELECT ST_AsGML({version},ST_GeomFromText('{wkt}',4326))"
                for rec in conn.execute(sa.text(sql)):
                    postgis_xml = rec[0]

                shape = gws.base.shape.from_wkt(wkt, gws.lib.crs.WGS84)
                el = gws.lib.gml.shape_to_element(
                    shape,
                    version=version,
                    coordinate_precision=0,
                    # NB postgis always uses EPSG
                    crs_format=gws.CrsFormat.epsg,
                )
                our_xml = el.to_string()

                # NB postgis does not write these attributes
                our_xml = _remove_attr(our_xml, 'decimal')
                our_xml = _remove_attr(our_xml, 'cs')
                our_xml = _remove_attr(our_xml, 'ts')
                
                # NB postgis does not write srsName on inner geometries, we do
                our_xml = _remove_attr(our_xml, 'srsName')
                postgis_xml = _remove_attr(postgis_xml, 'srsName')
                
                assert u.fxml(our_xml) == u.fxml(postgis_xml)


def _remove_attr(xml, name):
    return re.sub(r'\s+' + re.escape(name) + r'="[^"]*"', '', xml)