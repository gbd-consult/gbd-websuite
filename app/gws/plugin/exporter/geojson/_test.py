import json
import os
import zipfile

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.zipx
import gws.test.util as u


def _point(x, y):
    return gws.base.shape.from_xy(x, y, gws.lib.crs.WGS84)


def _polygon(coords):
    wkt = 'POLYGON((' + ', '.join(f'{x} {y}' for x, y in coords) + '))'
    return gws.base.shape.from_wkt(wkt, gws.lib.crs.WGS84)


@u.fixture(scope='module')
def root():
    u.pg.create('geojson_poi', {
        'id': 'int primary key',
        'name': 'text',
        'category': 'text',
        'geom': 'geometry(Point, 4326)',
    })
    u.pg.create('geojson_district', {
        'id': 'int primary key',
        'name': 'text',
        'area_km': 'float8',
        'geom': 'geometry(Polygon, 4326)',
    })
    u.pg.create('geojson_note', {
        'id': 'int primary key',
        'title': 'text',
        'body': 'text',
    })

    cfg = '''
        projects+ {
            uid "geojson_project"
            title "GeoJSON Test Project"
            access "allow all"

            models+ {
                uid "GEOJSON_POI"
                type "postgres"
                tableName "geojson_poi"
                fields+ { name "id"       type "integer" }
                fields+ { name "name"     type "text" }
                fields+ { name "category" type "text" }
                fields+ { name "geom"     type geometry geometryType "Point"   crs "EPSG:4326" }
            }

            models+ {
                uid "GEOJSON_DISTRICT"
                type "postgres"
                tableName "geojson_district"
                fields+ { name "id"      type "integer" }
                fields+ { name "name"    type "text" }
                fields+ { name "area_km" type "float" }
                fields+ { name "geom"    type geometry geometryType "Polygon" crs "EPSG:4326" }
            }

            models+ {
                uid "GEOJSON_NOTE"
                type "postgres"
                tableName "geojson_note"
                fields+ { name "id"    type "integer" }
                fields+ { name "title" type "text" }
                fields+ { name "body"  type "text" }
            }

            exporters+ {
                uid "EXP_1"
                type "geojson"
                title "GeoJSON Exporter"
                target "download"
                access "allow all"
            }

            exporters+ {
                uid "EXP_2"
                type "geojson"
                title "GeoJSON Exporter No Geometry"
                target "download"
                access "allow all"
                withNoGeometry true
            }
        }
    '''

    yield u.gws_root(cfg)


def _run_export(root, features, out_path, exporter_uid='EXP_1'):
    r = gws.ExportRequest(
        exporterUid=exporter_uid,
        projectUid='geojson_project',
        type=gws.ExportRequestType.vector,
        features=features,
    )
    root.app.exporterMgr.exec_export(r, out_path)


def _read_geojson(path):
    """Read GeoJSON text from a plain file or the first matching file in a zip."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith('.json'))
            return zf.read(name).decode('utf-8')
    with open(path) as f:
        return f.read()


def _read_geojson_from_zip(path, table_name):
    """Read GeoJSON text for a specific table from a zip."""
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith('.json') and table_name in n)
        return zf.read(name).decode('utf-8')


def _feature_names(gj):
    return [f['properties']['name'] for f in gj['features']]


##


def test_export_poi(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    p2 = _point(9.1, 52.0)

    u.pg.insert('geojson_poi', [
        {'id': 1, 'name': 'Restaurant Alpha', 'category': 'food',    'geom': p1.to_ewkb_hex()},
        {'id': 2, 'name': 'Museum Beta',       'category': 'culture', 'geom': p2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'poi_export.geojson')
    _run_export(root, [
        gws.FeatureProps(modelUid='GEOJSON_POI', uid='1'),
        gws.FeatureProps(modelUid='GEOJSON_POI', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)
    gj = json.loads(_read_geojson(out_path))

    assert gj['type'] == 'FeatureCollection'
    names = _feature_names(gj)
    assert 'Restaurant Alpha' in names
    assert 'Museum Beta' in names

    geom_types = {f['geometry']['type'] for f in gj['features']}
    assert 'Point' in geom_types

    categories = {f['properties']['category'] for f in gj['features']}
    assert 'food' in categories
    assert 'culture' in categories


def test_export_district(root: gws.Root, tmp_path):
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])
    d2 = _polygon([(9.0, 51.0), (10.0, 51.0), (10.0, 52.0), (9.0, 52.0), (9.0, 51.0)])

    u.pg.insert('geojson_district', [
        {'id': 1, 'name': 'North District', 'area_km': 12.5, 'geom': d1.to_ewkb_hex()},
        {'id': 2, 'name': 'South District', 'area_km': 8.3,  'geom': d2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'district_export.geojson')
    _run_export(root, [
        gws.FeatureProps(modelUid='GEOJSON_DISTRICT', uid='1'),
        gws.FeatureProps(modelUid='GEOJSON_DISTRICT', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)
    gj = json.loads(_read_geojson(out_path))

    assert gj['type'] == 'FeatureCollection'
    names = _feature_names(gj)
    assert 'North District' in names
    assert 'South District' in names

    geom_types = {f['geometry']['type'] for f in gj['features']}
    assert 'Polygon' in geom_types


def test_export_both_models(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])

    u.pg.insert('geojson_poi', [
        {'id': 10, 'name': 'POI One', 'category': 'tourism', 'geom': p1.to_ewkb_hex()},
    ])
    u.pg.insert('geojson_district', [
        {'id': 10, 'name': 'District One', 'area_km': 5.0, 'geom': d1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'both_export.geojson')
    _run_export(root, [
        gws.FeatureProps(modelUid='GEOJSON_POI',      uid='10'),
        gws.FeatureProps(modelUid='GEOJSON_DISTRICT',  uid='10'),
    ], out_path)

    assert os.path.isfile(out_path)

    # supportsMultiLayer=False: two models always produce separate files in a zip
    assert zipfile.is_zipfile(out_path)
    with zipfile.ZipFile(out_path) as zf:
        geojson_names = [n for n in zf.namelist() if n.endswith('.json')]
    assert len(geojson_names) == 2

    poi_gj = json.loads(_read_geojson_from_zip(out_path, 'geojson_poi'))
    district_gj = json.loads(_read_geojson_from_zip(out_path, 'geojson_district'))

    assert poi_gj['type'] == 'FeatureCollection'
    assert district_gj['type'] == 'FeatureCollection'

    poi_names = _feature_names(poi_gj)
    assert 'POI One' in poi_names
    assert 'District One' not in poi_names

    district_names = _feature_names(district_gj)
    assert 'District One' in district_names
    assert 'POI One' not in district_names


def test_geometry_coordinates_are_correct(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)

    u.pg.insert('geojson_poi', [
        {'id': 20, 'name': 'Coord Check', 'category': 'test', 'geom': p1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'coord_check.geojson')
    _run_export(root, [
        gws.FeatureProps(modelUid='GEOJSON_POI', uid='20'),
    ], out_path)

    gj = json.loads(_read_geojson(out_path))
    feature = next(f for f in gj['features'] if f['properties']['name'] == 'Coord Check')

    coords = feature['geometry']['coordinates']
    assert abs(coords[0] - 8.5) < 0.001
    assert abs(coords[1] - 51.2) < 0.001


def test_no_geometry_excluded_by_exp1(root: gws.Root, tmp_path):
    u.pg.insert('geojson_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_exp1.geojson')
    with u.raises(gws.Error):
        _run_export(root, [
            gws.FeatureProps(modelUid='GEOJSON_NOTE', uid='1'),
            gws.FeatureProps(modelUid='GEOJSON_NOTE', uid='2'),
        ], out_path, exporter_uid='EXP_1')


def test_no_geometry_included_by_exp2(root: gws.Root, tmp_path):
    u.pg.insert('geojson_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_exp2.geojson')
    _run_export(root, [
        gws.FeatureProps(modelUid='GEOJSON_NOTE', uid='1'),
        gws.FeatureProps(modelUid='GEOJSON_NOTE', uid='2'),
    ], out_path, exporter_uid='EXP_2')

    assert os.path.isfile(out_path)
    gj = json.loads(_read_geojson(out_path))

    assert gj['type'] == 'FeatureCollection'
    titles = {f['properties']['title'] for f in gj['features']}
    assert 'Note One' in titles
    assert 'Note Two' in titles
