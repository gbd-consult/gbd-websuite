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
    u.pg.create('kml_poi', {
        'id': 'int primary key',
        'name': 'text',
        'category': 'text',
        'geom': 'geometry(Point, 4326)',
    })
    u.pg.create('kml_district', {
        'id': 'int primary key',
        'name': 'text',
        'area_km': 'float8',
        'geom': 'geometry(Polygon, 4326)',
    })
    u.pg.create('kml_note', {
        'id': 'int primary key',
        'title': 'text',
        'body': 'text',
    })

    cfg = '''
        projects+ {
            uid "kml_project"
            title "KML Test Project"
            access "allow all"

            models+ {
                uid "KML_POI"
                type "postgres"
                tableName "kml_poi"
                fields+ { name "id"       type "integer" }
                fields+ { name "name"     type "text" }
                fields+ { name "category" type "text" }
                fields+ { name "geom"     type geometry geometryType "Point"   crs "EPSG:4326" }
            }

            models+ {
                uid "KML_DISTRICT"
                type "postgres"
                tableName "kml_district"
                fields+ { name "id"      type "integer" }
                fields+ { name "name"    type "text" }
                fields+ { name "area_km" type "float" }
                fields+ { name "geom"    type geometry geometryType "Polygon" crs "EPSG:4326" }
            }

            models+ {
                uid "KML_NOTE"
                type "postgres"
                tableName "kml_note"
                fields+ { name "id"    type "integer" }
                fields+ { name "title" type "text" }
                fields+ { name "body"  type "text" }
            }

            exporters+ {
                uid "EXP_1"
                type "kml"
                title "KML Exporter"
                target "download"
                access "allow all"
            }

            exporters+ {
                uid "EXP_2"
                type "kml"
                title "KML Exporter No Geometry"
                target "download"
                access "allow all"
                withNoGeometry true
            }

            exporters+ {
                uid "EXP_3"
                type "kml"
                title "KML Exporter Split"
                target "download"
                access "allow all"
                withSplitLayers true
            }
        }
    '''

    yield u.gws_root(cfg)


def _run_export(root, features, out_path, exporter_uid='EXP_1'):
    r = gws.ExportRequest(
        exporterUid=exporter_uid,
        projectUid='kml_project',
        type=gws.ExportRequestType.vector,
        features=features,
    )
    root.app.exporterMgr.exec_export(r, out_path)


def _read_kml(path):
    with open(path) as f:
        return f.read()


def _read_kml_from_zip(zip_path, kml_name):
    with u.temp_dir_in_base_dir() as tmp_dir:
        gws.lib.zipx.unzip_path(zip_path, tmp_dir, flat=True)
        return _read_kml(os.path.join(tmp_dir, kml_name))


##


def test_export_poi(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    p2 = _point(9.1, 52.0)

    u.pg.insert('kml_poi', [
        {'id': 1, 'name': 'Restaurant Alpha', 'category': 'food',    'geom': p1.to_ewkb_hex()},
        {'id': 2, 'name': 'Museum Beta',       'category': 'culture', 'geom': p2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'poi_export.kml')
    _run_export(root, [
        gws.FeatureProps(modelUid='KML_POI', uid='1'),
        gws.FeatureProps(modelUid='KML_POI', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_kml(out_path), nl=False)

    assert '<name>Restaurant Alpha</name>' in xml
    assert '<name>Museum Beta</name>' in xml
    assert '<SimpleData name="category">food</SimpleData>' in xml
    assert '<SimpleData name="category">culture</SimpleData>' in xml
    assert '<Point>' in xml


def test_export_district(root: gws.Root, tmp_path):
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])
    d2 = _polygon([(9.0, 51.0), (10.0, 51.0), (10.0, 52.0), (9.0, 52.0), (9.0, 51.0)])

    u.pg.insert('kml_district', [
        {'id': 1, 'name': 'North District', 'area_km': 12.5, 'geom': d1.to_ewkb_hex()},
        {'id': 2, 'name': 'South District', 'area_km': 8.3,  'geom': d2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'district_export.kml')
    _run_export(root, [
        gws.FeatureProps(modelUid='KML_DISTRICT', uid='1'),
        gws.FeatureProps(modelUid='KML_DISTRICT', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_kml(out_path), nl=False)

    assert '<name>North District</name>' in xml
    assert '<name>South District</name>' in xml
    assert '<Polygon>' in xml


def test_export_both_models_single_file(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])

    u.pg.insert('kml_poi', [
        {'id': 10, 'name': 'POI One', 'category': 'tourism', 'geom': p1.to_ewkb_hex()},
    ])
    u.pg.insert('kml_district', [
        {'id': 10, 'name': 'District One', 'area_km': 5.0, 'geom': d1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'both_export.kml')
    _run_export(root, [
        gws.FeatureProps(modelUid='KML_POI',      uid='10'),
        gws.FeatureProps(modelUid='KML_DISTRICT',  uid='10'),
    ], out_path)

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_kml(out_path), nl=False)

    # both layer folders are present
    assert '<name>kml_poi</name>' in xml
    assert '<name>kml_district</name>' in xml

    # both feature names are present
    assert '<name>POI One</name>' in xml
    assert '<name>District One</name>' in xml

    # both geometry types are present
    assert '<Point>' in xml
    assert '<Polygon>' in xml


def test_export_split_layers(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])

    u.pg.insert('kml_poi', [
        {'id': 20, 'name': 'Split POI', 'category': 'test', 'geom': p1.to_ewkb_hex()},
    ])
    u.pg.insert('kml_district', [
        {'id': 20, 'name': 'Split District', 'area_km': 3.0, 'geom': d1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'split_export.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='KML_POI',     uid='20'),
        gws.FeatureProps(modelUid='KML_DISTRICT', uid='20'),
    ], out_path, exporter_uid='EXP_3')

    assert os.path.isfile(out_path)

    with zipfile.ZipFile(out_path) as zf:
        names_in_zip = set(zf.namelist())

    assert 'kml_poi.kml' in names_in_zip
    assert 'kml_district.kml' in names_in_zip

    poi_xml = u.fxml(_read_kml_from_zip(out_path, 'kml_poi.kml'), nl=False)
    assert '<name>Split POI</name>' in poi_xml
    assert '<Point>' in poi_xml
    assert 'Split District' not in poi_xml

    district_xml = u.fxml(_read_kml_from_zip(out_path, 'kml_district.kml'), nl=False)
    assert '<name>Split District</name>' in district_xml
    assert '<Polygon>' in district_xml
    assert 'Split POI' not in district_xml


def test_no_geometry_excluded_by_exp1(root: gws.Root, tmp_path):
    u.pg.insert('kml_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_exp1.kml')
    with u.raises(gws.Error):
        _run_export(root, [
            gws.FeatureProps(modelUid='KML_NOTE', uid='1'),
            gws.FeatureProps(modelUid='KML_NOTE', uid='2'),
        ], out_path, exporter_uid='EXP_1')


def test_no_geometry_included_by_exp2(root: gws.Root, tmp_path):
    u.pg.insert('kml_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_exp2.kml')
    _run_export(root, [
        gws.FeatureProps(modelUid='KML_NOTE', uid='1'),
        gws.FeatureProps(modelUid='KML_NOTE', uid='2'),
    ], out_path, exporter_uid='EXP_2')

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_kml(out_path), nl=False)

    assert '<SimpleData name="title">Note One</SimpleData>' in xml
    assert '<SimpleData name="title">Note Two</SimpleData>' in xml
    assert '<SimpleData name="body">Body one</SimpleData>' in xml
    assert '<SimpleData name="body">Body two</SimpleData>' in xml
