import csv
import io
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
    u.pg.create('csv_poi', {
        'id': 'int primary key',
        'name': 'text',
        'category': 'text',
        'geom': 'geometry(Point, 4326)',
    })
    u.pg.create('csv_district', {
        'id': 'int primary key',
        'name': 'text',
        'area_km': 'float8',
        'geom': 'geometry(Polygon, 4326)',
    })
    u.pg.create('csv_note', {
        'id': 'int primary key',
        'title': 'text',
        'body': 'text',
    })

    cfg = '''
        projects+ {
            uid "csv_project"
            title "CSV Test Project"
            access "allow all"

            models+ {
                uid "CSV_POI"
                type "postgres"
                tableName "csv_poi"
                fields+ { name "id"       type "integer" }
                fields+ { name "name"     type "text" }
                fields+ { name "category" type "text" }
                fields+ { name "geom"     type geometry geometryType "Point"   crs "EPSG:4326" }
            }

            models+ {
                uid "CSV_DISTRICT"
                type "postgres"
                tableName "csv_district"
                fields+ { name "id"      type "integer" }
                fields+ { name "name"    type "text" }
                fields+ { name "area_km" type "float" }
                fields+ { name "geom"    type geometry geometryType "Polygon" crs "EPSG:4326" }
            }

            models+ {
                uid "CSV_NOTE"
                type "postgres"
                tableName "csv_note"
                fields+ { name "id"    type "integer" }
                fields+ { name "title" type "text" }
                fields+ { name "body"  type "text" }
            }

            exporters+ {
                uid "EXP_1"
                type "csv"
                title "CSV Exporter"
                target "download"
                access "allow all"
            }
        }
    '''

    yield u.gws_root(cfg)


def _run_export(root, features, out_path, exporter_uid='EXP_1'):
    r = gws.ExportRequest(
        exporterUid=exporter_uid,
        projectUid='csv_project',
        type=gws.ExportRequestType.vector,
        features=features,
    )
    root.app.exporterMgr.exec_export(r, out_path)


def _read_csv_from_zip(zip_path, csv_name):
    with u.temp_dir_in_base_dir() as tmp_dir:
        gws.lib.zipx.unzip_path(zip_path, tmp_dir, flat=True)
        with open(os.path.join(tmp_dir, csv_name), newline='') as f:
            return f.read()


def _parse_csv(text):
    return list(csv.DictReader(io.StringIO(text)))


##


def test_export_poi_produces_zip(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)

    u.pg.insert('csv_poi', [
        {'id': 1, 'name': 'Restaurant Alpha', 'category': 'food', 'geom': p1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'poi_export.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='CSV_POI', uid='1'),
    ], out_path)

    assert os.path.isfile(out_path)
    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
    assert any(n.endswith('.csv') for n in names)


def test_export_poi_attributes(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    p2 = _point(9.1, 52.0)

    u.pg.insert('csv_poi', [
        {'id': 2, 'name': 'Restaurant Alpha', 'category': 'food',    'geom': p1.to_ewkb_hex()},
        {'id': 3, 'name': 'Museum Beta',       'category': 'culture', 'geom': p2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'poi_attrs.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='CSV_POI', uid='2'),
        gws.FeatureProps(modelUid='CSV_POI', uid='3'),
    ], out_path)

    with zipfile.ZipFile(out_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith('.csv') and 'csv_poi' in n)

    text = _read_csv_from_zip(out_path, csv_name)
    rows = _parse_csv(text)

    names = {r['name'] for r in rows}
    categories = {r['category'] for r in rows}

    assert 'Restaurant Alpha' in names
    assert 'Museum Beta' in names
    assert 'food' in categories
    assert 'culture' in categories


def test_export_district_attributes(root: gws.Root, tmp_path):
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])
    d2 = _polygon([(9.0, 51.0), (10.0, 51.0), (10.0, 52.0), (9.0, 52.0), (9.0, 51.0)])

    u.pg.insert('csv_district', [
        {'id': 1, 'name': 'North District', 'area_km': 12.5, 'geom': d1.to_ewkb_hex()},
        {'id': 2, 'name': 'South District', 'area_km': 8.3,  'geom': d2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'district_attrs.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='CSV_DISTRICT', uid='1'),
        gws.FeatureProps(modelUid='CSV_DISTRICT', uid='2'),
    ], out_path)

    with zipfile.ZipFile(out_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith('.csv') and 'csv_district' in n)

    text = _read_csv_from_zip(out_path, csv_name)
    rows = _parse_csv(text)

    names = {r['name'] for r in rows}
    assert 'North District' in names
    assert 'South District' in names


def test_split_layers_always_applied_mixed_models(root: gws.Root, tmp_path):
    """withSplitLayers is hardcoded True: mixed models always produce separate CSVs in a zip."""
    p1 = _point(8.5, 51.2)
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])

    u.pg.insert('csv_poi', [
        {'id': 10, 'name': 'Mixed POI', 'category': 'test', 'geom': p1.to_ewkb_hex()},
    ])
    u.pg.insert('csv_district', [
        {'id': 10, 'name': 'Mixed District', 'area_km': 5.0, 'geom': d1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'mixed_export.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='CSV_POI',      uid='10'),
        gws.FeatureProps(modelUid='CSV_DISTRICT',  uid='10'),
    ], out_path)

    assert os.path.isfile(out_path)

    with zipfile.ZipFile(out_path) as zf:
        names_in_zip = set(zf.namelist())

    assert any(n.endswith('.csv') and 'csv_poi' in n for n in names_in_zip), \
        f'no csv_poi csv in {names_in_zip}'
    assert any(n.endswith('.csv') and 'csv_district' in n for n in names_in_zip), \
        f'no csv_district csv in {names_in_zip}'

    poi_name = next(n for n in names_in_zip if n.endswith('.csv') and 'csv_poi' in n)
    district_name = next(n for n in names_in_zip if n.endswith('.csv') and 'csv_district' in n)

    poi_rows = _parse_csv(_read_csv_from_zip(out_path, poi_name))
    district_rows = _parse_csv(_read_csv_from_zip(out_path, district_name))

    poi_names = {r['name'] for r in poi_rows}
    district_names = {r['name'] for r in district_rows}

    assert 'Mixed POI' in poi_names
    assert 'Mixed District' not in poi_names

    assert 'Mixed District' in district_names
    assert 'Mixed POI' not in district_names


def test_no_geometry_model_works(root: gws.Root, tmp_path):
    """withNoGeometry is hardcoded True: geometry-free models are exported without error."""
    u.pg.insert('csv_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_export.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='CSV_NOTE', uid='1'),
        gws.FeatureProps(modelUid='CSV_NOTE', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)

    with zipfile.ZipFile(out_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith('.csv'))

    text = _read_csv_from_zip(out_path, csv_name)
    rows = _parse_csv(text)

    titles = {r['title'] for r in rows}
    bodies = {r['body'] for r in rows}

    assert 'Note One' in titles
    assert 'Note Two' in titles
    assert 'Body one' in bodies
    assert 'Body two' in bodies


def test_geometry_field_not_in_csv_columns(root: gws.Root, tmp_path):
    """Geometry model fields are not exported as CSV columns."""
    p1 = _point(8.5, 51.2)

    u.pg.insert('csv_poi', [
        {'id': 20, 'name': 'Geom Check', 'category': 'test', 'geom': p1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'geom_check.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='CSV_POI', uid='20'),
    ], out_path)

    with zipfile.ZipFile(out_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith('.csv') and 'csv_poi' in n)

    text = _read_csv_from_zip(out_path, csv_name)
    rows = _parse_csv(text)

    assert rows
    assert 'geom' not in rows[0]
