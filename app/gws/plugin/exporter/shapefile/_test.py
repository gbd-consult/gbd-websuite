import os
import zipfile

import gws
import gws.base.shape
import gws.lib.gdalx
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
    u.pg.create('shp_poi', {
        'id': 'int primary key',
        'name': 'text',
        'category': 'text',
        'geom': 'geometry(Point, 4326)',
    })
    u.pg.create('shp_district', {
        'id': 'int primary key',
        'name': 'text',
        'area_km': 'float8',
        'geom': 'geometry(Polygon, 4326)',
    })
    u.pg.create('shp_note', {
        'id': 'int primary key',
        'title': 'text',
        'body': 'text',
    })

    cfg = '''
        projects+ {
            uid "test_project"
            title "Test Project"
            access "allow all"

            models+ {
                uid "SHP_POI"
                type "postgres"
                tableName "shp_poi"
                fields+ { name "id"       type "integer" }
                fields+ { name "name"     type "text" }
                fields+ { name "category" type "text" }
                fields+ { name "geom"     type geometry geometryType "Point"   crs "EPSG:4326" }
            }

            models+ {
                uid "SHP_DISTRICT"
                type "postgres"
                tableName "shp_district"
                fields+ { name "id"      type "integer" }
                fields+ { name "name"    type "text" }
                fields+ { name "area_km" type "float" }
                fields+ { name "geom"    type geometry geometryType "Polygon" crs "EPSG:4326" }
            }

            models+ {
                uid "SHP_NOTE"
                type "postgres"
                tableName "shp_note"
                fields+ { name "id"    type "integer" }
                fields+ { name "title" type "text" }
                fields+ { name "body"  type "text" }
            }

            exporters+ {
                uid "EXP_1"
                type "shapefile"
                title "Shapefile Exporter"
                target "download"
                access "allow all"
            }

            exporters+ {
                uid "EXP_2"
                type "shapefile"
                title "Shapefile Exporter No Geometry"
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
        projectUid='test_project',
        type=gws.ExportRequestType.vector,
        features=features,
    )
    root.app.exporterMgr.exec_export(r, out_path)


def _read_shapefile(zip_path, shp_name):
    """Unzip export archive and return all records from the named shapefile."""
    with u.temp_dir_in_base_dir() as tmp_dir:
        gws.lib.zipx.unzip_path(zip_path, tmp_dir, flat=True)
        shp_path = os.path.join(tmp_dir, shp_name)
        with gws.lib.gdalx.open_vector(shp_path) as ds:
            la = ds.layer(0)
            return la.get_all()


def test_export_poi(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    p2 = _point(9.1, 52.0)

    u.pg.insert('shp_poi', [
        {'id': 1, 'name': 'Restaurant Alpha', 'category': 'food',    'geom': p1.to_ewkb_hex()},
        {'id': 2, 'name': 'Museum Beta',       'category': 'culture', 'geom': p2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'poi_export.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='SHP_POI', uid='1'),
        gws.FeatureProps(modelUid='SHP_POI', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)

    records = _read_shapefile(out_path, 'shp_poi.shp')
    assert len(records) == 2

    names = {r.attributes['name'] for r in records}
    assert names == {'Restaurant Alpha', 'Museum Beta'}

    categories = {r.attributes['category'] for r in records}
    assert categories == {'food', 'culture'}

    for rec in records:
        assert rec.shape is not None


def test_export_district(root: gws.Root, tmp_path):
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])
    d2 = _polygon([(9.0, 51.0), (10.0, 51.0), (10.0, 52.0), (9.0, 52.0), (9.0, 51.0)])

    u.pg.insert('shp_district', [
        {'id': 1, 'name': 'North District', 'area_km': 12.5, 'geom': d1.to_ewkb_hex()},
        {'id': 2, 'name': 'South District', 'area_km': 8.3,  'geom': d2.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'district_export.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='SHP_DISTRICT', uid='1'),
        gws.FeatureProps(modelUid='SHP_DISTRICT', uid='2'),
    ], out_path)

    assert os.path.isfile(out_path)

    records = _read_shapefile(out_path, 'shp_district.shp')
    assert len(records) == 2

    names = {r.attributes['name'] for r in records}
    assert names == {'North District', 'South District'}

    for rec in records:
        assert rec.shape is not None


def test_export_both_models(root: gws.Root, tmp_path):
    p1 = _point(8.5, 51.2)
    d1 = _polygon([(7.0, 50.0), (8.0, 50.0), (8.0, 51.0), (7.0, 51.0), (7.0, 50.0)])

    u.pg.insert('shp_poi', [
        {'id': 10, 'name': 'POI One', 'category': 'tourism', 'geom': p1.to_ewkb_hex()},
    ])
    u.pg.insert('shp_district', [
        {'id': 10, 'name': 'District One', 'area_km': 5.0, 'geom': d1.to_ewkb_hex()},
    ])

    out_path = str(tmp_path / 'both_export.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='SHP_POI',      uid='10'),
        gws.FeatureProps(modelUid='SHP_DISTRICT',  uid='10'),
    ], out_path)

    assert os.path.isfile(out_path)

    with zipfile.ZipFile(out_path) as zf:
        names_in_zip = {n for n in zf.namelist() if n.endswith('.shp')}

    assert 'shp_poi.shp' in names_in_zip
    assert 'shp_district.shp' in names_in_zip

    poi_records = _read_shapefile(out_path, 'shp_poi.shp')
    assert len(poi_records) == 1
    assert poi_records[0].attributes['name'] == 'POI One'

    district_records = _read_shapefile(out_path, 'shp_district.shp')
    assert len(district_records) == 1
    assert district_records[0].attributes['name'] == 'District One'


def test_no_geometry_excluded_by_exp1(root: gws.Root, tmp_path):
    u.pg.insert('shp_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_exp1.zip')
    with u.raises(gws.Error):
        _run_export(root, [
            gws.FeatureProps(modelUid='SHP_NOTE', uid='1'),
            gws.FeatureProps(modelUid='SHP_NOTE', uid='2'),
        ], out_path, exporter_uid='EXP_1')


def test_no_geometry_included_by_exp2(root: gws.Root, tmp_path):
    u.pg.insert('shp_note', [
        {'id': 1, 'title': 'Note One', 'body': 'Body one'},
        {'id': 2, 'title': 'Note Two', 'body': 'Body two'},
    ])

    out_path = str(tmp_path / 'note_exp2.zip')
    _run_export(root, [
        gws.FeatureProps(modelUid='SHP_NOTE', uid='1'),
        gws.FeatureProps(modelUid='SHP_NOTE', uid='2'),
    ], out_path, exporter_uid='EXP_2')

    assert os.path.isfile(out_path)

    records = _read_shapefile(out_path, 'shp_note.shp')
    assert len(records) == 2

    titles = {r.attributes['title'] for r in records}
    assert titles == {'Note One', 'Note Two'}
