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
    u.pg.create('gml_poi', {
        'id': 'int primary key',
        'name': 'text',
        'category': 'text',
        'geom': 'geometry(Point, 4326)',
    })
    u.pg.create('gml_district', {
        'id': 'int primary key',
        'name': 'text',
        'area_km': 'float8',
        'geom': 'geometry(Polygon, 4326)',
    })
    u.pg.create('gml_note', {
        'id': 'int primary key',
        'title': 'text',
        'body': 'text',
    })

    cfg = '''
        projects+ {
            uid "gml_project"
            title "GML Test Project"
            access "allow all"

            models+ {
                uid "GML_POI"
                type "postgres"
                tableName "gml_poi"
                fields+ { name "id"       type "integer" }
                fields+ { name "name"     type "text" }
                fields+ { name "category" type "text" }
                fields+ { name "geom"     type geometry geometryType "Point"   crs "EPSG:4326" }
            }

            models+ {
                uid "GML_DISTRICT"
                type "postgres"
                tableName "gml_district"
                fields+ { name "id"      type "integer" }
                fields+ { name "name"    type "text" }
                fields+ { name "area_km" type "float" }
                fields+ { name "geom"    type geometry geometryType "Polygon" crs "EPSG:4326" }
            }

            models+ {
                uid "GML_NOTE"
                type "postgres"
                tableName "gml_note"
                fields+ { name "id"    type "integer" }
                fields+ { name "title" type "text" }
                fields+ { name "body"  type "text" }
            }

            exporters+ {
                uid "EXP_1"
                type "gml"
                title "GML Exporter"
                target "download"
                access "allow all"
            }

            exporters+ {
                uid "EXP_2"
                type "gml"
                title "GML Exporter No Geometry"
                target "download"
                access "allow all"
                withNoGeometry true
            }

            exporters+ {
                uid "EXP_3"
                type "gml"
                title "GML Exporter Multi Layer"
                target "download"
                access "allow all"
                withMultiLayer true
            }

            exporters+ {
                uid "EXP_4"
                type "gml"
                title "GML2 Exporter with prefix"
                target "download"
                access "allow all"
                options {
                    FORMAT "GML2"
                    PREFIX "zzz"
                }
            }
        }
    '''

    yield u.gws_root(cfg)


def _run_export(root, features, out_path, exporter_uid='EXP_1'):
    r = gws.ExportRequest(
        exporterUid=exporter_uid,
        projectUid='gml_project',
        type=gws.ExportRequestType.vector,
        features=features,
    )
    root.app.exporterMgr.exec_export(r, out_path)


def _read_gml(path):
    if zipfile.is_zipfile(path):
        parts = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                parts.append(zf.read(name).decode('utf-8'))
        return '\n'.join(parts)
    with open(path) as f:
        return f.read()


##


def test_export_poi(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    p2 = _point(9.1, 52.0)

    u.pg.insert('gml_poi', [
        {'id': 1, 'name': 'Restaurant Alpha', 'category': 'food',    'geom': p1.to_ewkb_hex()},
        {'id': 2, 'name': 'Museum Beta',       'category': 'culture', 'geom': p2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'poi_export.gml')
    _run_export(root, [
        gws.FeatureProps(modelUid='GML_POI', uid='1'),
        gws.FeatureProps(modelUid='GML_POI', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_gml(out_path), nl=False)

    assert 'Restaurant Alpha' in xml
    assert 'Museum Beta' in xml
    assert 'food' in xml
    assert 'culture' in xml
    assert 'Point' in xml


def test_export_district(root: gws.Root, tmp_path):
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])
    d2 = _polygon([(9.0, 51.0), (10.0, 51.0), (10.0, 52.0), (9.0, 52.0), (9.0, 51.0)])

    u.pg.insert('gml_district', [
        {'id': 1, 'name': 'North District', 'area_km': 12.5, 'geom': d1.to_ewkb_hex()},
        {'id': 2, 'name': 'South District', 'area_km': 8.3,  'geom': d2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'district_export.gml')
    _run_export(root, [
        gws.FeatureProps(modelUid='GML_DISTRICT', uid='1'),
        gws.FeatureProps(modelUid='GML_DISTRICT', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_gml(out_path), nl=False)

    assert 'North District' in xml
    assert 'South District' in xml
    assert 'Polygon' in xml


def test_export_both_models(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])

    u.pg.insert('gml_poi', [
        {'id': 10, 'name': 'POI One', 'category': 'tourism', 'geom': p1.to_ewkb_hex()},
    ])
    u.pg.insert('gml_district', [
        {'id': 10, 'name': 'District One', 'area_km': 5.0, 'geom': d1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'both_export.gml')
    _run_export(root, [
        gws.FeatureProps(modelUid='GML_POI',      uid='10'),
        gws.FeatureProps(modelUid='GML_DISTRICT',  uid='10'),
    ], out_path)

    assert os.path.isfile(out_path)

    # withMultiLayer false (default): each model in its own .gml file, packed into a zip
    if zipfile.is_zipfile(out_path):
        with zipfile.ZipFile(out_path) as zf:
            gml_names = [n for n in zf.namelist() if n.endswith('.gml')]
        assert len(gml_names) == 2, f'expected 2 .gml files in zip, got {gml_names}'

    text = _read_gml(out_path)
    assert 'POI One' in text
    assert 'District One' in text
    assert 'Point' in text
    assert 'Polygon' in text


def test_multi_layer_single_model(root: gws.Root, tmp_path):
    p1 = _point(7.5, 50.5)

    u.pg.insert('gml_poi', [
        {'id': 40, 'name': 'ML POI', 'category': 'test', 'geom': p1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'ml_single.gml')
    _run_export(root, [
        gws.FeatureProps(modelUid='GML_POI', uid='40'),
    ], out_path, exporter_uid='EXP_3')

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_gml(out_path), nl=False)
    assert 'ML POI' in xml
    assert 'Point' in xml


def test_multi_layer_both_models_single_file(root: gws.Root, tmp_path):
    p1 = _point(7.5, 50.5)
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])

    u.pg.insert('gml_poi', [
        {'id': 50, 'name': 'ML POI Two', 'category': 'test', 'geom': p1.to_ewkb_hex()},
    ])
    u.pg.insert('gml_district', [
        {'id': 50, 'name': 'ML District Two', 'area_km': 4.0, 'geom': d1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'ml_both.gml')
    _run_export(root, [
        gws.FeatureProps(modelUid='GML_POI',      uid='50'),
        gws.FeatureProps(modelUid='GML_DISTRICT',  uid='50'),
    ], out_path, exporter_uid='EXP_3')

    assert os.path.isfile(out_path)

    # withMultiLayer true: both models in a single .gml file (not split into separate files)
    if zipfile.is_zipfile(out_path):
        with zipfile.ZipFile(out_path) as zf:
            gml_names = [n for n in zf.namelist() if n.endswith('.gml')]
        assert len(gml_names) == 1, f'expected 1 .gml in zip, got {gml_names}'

    xml = u.fxml(_read_gml(out_path), nl=False)
    assert 'ML POI Two' in xml
    assert 'ML District Two' in xml
    assert 'Point' in xml
    assert 'Polygon' in xml


def test_no_geometry_excluded_by_exp1(root: gws.Root, tmp_path):
    u.pg.insert('gml_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_exp1.gml')
    with u.raises(gws.Error):
        _run_export(root, [
            gws.FeatureProps(modelUid='GML_NOTE', uid='1'),
            gws.FeatureProps(modelUid='GML_NOTE', uid='2'),
        ], out_path, exporter_uid='EXP_1')


def test_no_geometry_included_by_exp2(root: gws.Root, tmp_path):
    u.pg.insert('gml_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_exp2.gml')
    _run_export(root, [
        gws.FeatureProps(modelUid='GML_NOTE', uid='1'),
        gws.FeatureProps(modelUid='GML_NOTE', uid='2'),
    ], out_path, exporter_uid='EXP_2')

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_gml(out_path), nl=False)

    assert 'Note One' in xml
    assert 'Note Two' in xml
    assert 'Body one' in xml
    assert 'Body two' in xml


def test_gml2_with_prefix(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)

    u.pg.insert('gml_poi', [
        {'id': 30, 'name': 'Prefix POI', 'category': 'test', 'geom': p1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'prefix_export.gml')
    _run_export(root, [
        gws.FeatureProps(modelUid='GML_POI', uid='30'),
    ], out_path, exporter_uid='EXP_4')

    assert os.path.isfile(out_path)
    xml = u.fxml(_read_gml(out_path), nl=False)

    assert 'Prefix POI' in xml
    assert 'zzz' in xml
