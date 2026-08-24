"""Tests for the qfieldcloud packager."""

from typing import cast

import os

import gws
import gws.lib.gdalx
import gws.lib.jsonx
import gws.lib.xmlx
import gws.test.util as u

from gws.plugin.qfieldcloud import action as action_mod, packager
from gws.plugin.qfieldcloud._test import util as tu

CONFIG = """
    projects+ {
        uid "PROJECT_1"
        access "allow all"
        actions+ {
            type "qfieldcloud"
            uid "ACTION_1"
            access "allow all"
            projects+ {
                uid "QFC_1"
                title "QField Test"
                access "allow all"
                provider.path {QGS_PATH}
                models+ {
                    uid "MODEL_POI"
                    type "postgres"
                    tableName "qfc.poi"
                    isEditable true
                    permissions.edit "allow all"
                    fields+ { name "id"          type "integer" }
                    fields+ { name "name"        type "text" }
                    fields+ { name "district_id" type "integer" }
                    fields+ { name "geom"        type "geometry" }
                    fields+ {
                        type "file"
                        name "photo_file"
                        contentColumn "photo_content"
                        nameColumn "photo"
                    }
                }
            }
        }
    }
"""


@u.fixture(scope='module')
def root():
    tu.create_tables()
    yield u.gws_root(CONFIG, QGS_PATH=repr(tu.qgs_path('packager')))


def _act(root) -> action_mod.Object:
    return cast(action_mod.Object, root.get('ACTION_1'))


def _package(root, out_dir, **args) -> packager.Args:
    act = _act(root)
    qp = act.qfcProjects[0]

    pa = packager.Args(
        uid='TEST',
        qfcProject=qp,
        caps=act.get_caps(qp),
        project=root.app.project('PROJECT_1'),
        user=root.app.authMgr.systemUser,
        packageDir=str(out_dir),
        mapCacheDir=gws.u.ensure_dir(str(out_dir) + '/cache'),
        withBaseMap=False,
        withData=False,
        withMedia=False,
        withQgis=False,
    )
    pa.update(args)

    po = packager.Object()
    po.create_package(root, pa)
    return po


def _poi_rows():
    return [
        {'id': 1, 'name': 'inside 1', 'district_id': 10, 'photo': 'p1.jpg', 'geom': tu.point(750000, 6650000)},
        {'id': 2, 'name': 'inside 2', 'district_id': 10, 'photo': None, 'geom': tu.point(748000, 6648000)},
        {'id': 3, 'name': 'outside', 'district_id': 10, 'photo': None, 'geom': tu.point(700000, 6600000)},
    ]


##


def test_write_data(root: gws.Root, tmp_path):
    u.pg.insert('qfc.poi', _poi_rows())

    _package(root, tmp_path, withData=True)

    path = str(tmp_path / 'qm_qfc_poi.gpkg')
    assert os.path.isfile(path)

    with gws.lib.gdalx.open_vector(path) as ds:
        la = ds.require_layer('qm_qfc_poi')
        assert la.count() == 3
        recs = la.get_all()

    names = sorted(r.attributes['name'] for r in recs)
    assert names == ['inside 1', 'inside 2', 'outside']

    r = next(r for r in recs if r.attributes['name'] == 'inside 1')
    assert r.attributes['id'] == 1
    assert r.attributes['district_id'] == 10
    u.check.close((r.shape.x, r.shape.y), (750000, 6650000), abs_tol=0.001)


def test_write_data_skips_unsupported_field_types(root: gws.Root, tmp_path):
    u.pg.insert('qfc.poi', _poi_rows())

    _package(root, tmp_path, withData=True)

    with gws.lib.gdalx.open_vector(str(tmp_path / 'qm_qfc_poi.gpkg')) as ds:
        desc = ds.require_layer('qm_qfc_poi').describe()

    # the "file" field has no gpkg counterpart
    assert 'photo_file' not in desc.columnMap
    assert 'photo_content' not in desc.columnMap
    assert 'name' in desc.columnMap


def test_write_data_renames_the_fid_column(root: gws.Root, tmp_path):
    u.pg.insert('qfc.note', [
        {'id': 1, 'fid': 111, 'kind': 'a', 'text': 'note one', 'geom': tu.point(750000, 6650000)},
        {'id': 2, 'fid': 222, 'kind': 'b', 'text': 'note two', 'geom': tu.point(751000, 6651000)},
    ])

    _package(root, tmp_path, withData=True)

    with gws.lib.gdalx.open_vector(str(tmp_path / 'qm_qfc_note.gpkg')) as ds:
        la = ds.require_layer('qm_qfc_note')
        desc = la.describe()
        recs = la.get_all()

    assert 'fid_gws' in desc.columnMap

    fids = sorted(r.attributes['fid_gws'] for r in recs if r.attributes['fid_gws'] is not None)
    assert fids == [111, 222]


def test_write_data_writes_each_table_once(root: gws.Root, tmp_path):
    u.pg.insert('qfc.poi', _poi_rows())

    po = _package(root, tmp_path, withData=True)

    # removed_L7 uses the same table as poi_L1 but is not exported at all
    assert sorted(po.pathMap) == ['qm_qfc_district.gpkg', 'qm_qfc_note.gpkg', 'qm_qfc_poi.gpkg']


def test_write_data_with_area_of_interest(tmp_path):
    tu.create_tables()
    u.pg.insert('qfc.poi', _poi_rows())

    def patch(root_el):
        tu.set_project_prop(root_el, 'offlineCopyOnlyAoi', '1', 'int')

    root = u.gws_root(CONFIG, QGS_PATH=repr(tu.qgs_path('packager_aoi', patch)))
    _package(root, tmp_path, withData=True)

    with gws.lib.gdalx.open_vector(str(tmp_path / 'qm_qfc_poi.gpkg')) as ds:
        recs = ds.require_layer('qm_qfc_poi').get_all()

    assert sorted(r.attributes['name'] for r in recs) == ['inside 1', 'inside 2']


##


def test_write_media(root: gws.Root, tmp_path):
    d = gws.u.ensure_dir(f'{tu.WORK_DIR}/DCIM')
    gws.u.ensure_dir(f'{d}/sub')
    gws.u.write_file(f'{d}/a.jpg', 'a')
    gws.u.write_file(f'{d}/sub/b.jpg', 'b')

    po = _package(root, tmp_path, withMedia=True)

    assert po.pathMap['DCIM/a.jpg'] == f'{d}/a.jpg'
    assert po.pathMap['DCIM/sub/b.jpg'] == f'{d}/sub/b.jpg'


def test_write_media_skips_missing_dirs(root: gws.Root, tmp_path):
    # "/tmp/qfc_abs" is listed in the project but does not exist
    po = _package(root, tmp_path, withMedia=True)
    assert not any(p.startswith('/tmp/qfc_abs') for p in po.pathMap.values())


##


def test_write_qgis_project(root: gws.Root, tmp_path):
    po = _package(root, tmp_path, withQgis=True)

    path = str(tmp_path / 'QFC_1.qgs')
    assert os.path.isfile(path)
    assert os.path.isfile(path + '.source.qgs')
    assert po.pathMap['QFC_1.qgs'] == path

    root_el = gws.lib.xmlx.from_path(path)

    ids = [el.textof('id') for el in root_el.findall('.//maplayer')]
    assert 'poi_L1' in ids
    assert 'removed_L7' not in ids
    assert 'query_L4' not in ids

    for el in root_el.findall('.//maplayer'):
        if el.textof('id') == 'poi_L1':
            assert el.textof('datasource') == './qm_qfc_poi.gpkg|layername=qm_qfc_poi'
            assert el.textof('provider') == 'ogr'

    assert root_el.textof('properties/Paths/Absolute') == 'false'


def test_write_qgis_project_keeps_the_source(root: gws.Root, tmp_path):
    _package(root, tmp_path, withQgis=True)

    src = gws.u.read_file(str(tmp_path / 'QFC_1.qgs.source.qgs'))
    assert 'service=' in src
    assert 'removed_L7' in src


##


def test_create_package_writes_the_path_map(root: gws.Root, tmp_path):
    u.pg.insert('qfc.poi', _poi_rows())

    _package(root, tmp_path, withData=True, withQgis=True)

    assert gws.u.read_file(str(tmp_path / packager.COMPLETE_FILE)) == '1'

    pm = gws.lib.jsonx.from_path(str(tmp_path / packager.PATH_MAP_FILE))
    assert sorted(pm) == ['QFC_1.qgs', 'qm_qfc_district.gpkg', 'qm_qfc_note.gpkg', 'qm_qfc_poi.gpkg']
    for name, path in pm.items():
        assert os.path.isfile(path)


def test_create_package_from_cli(root: gws.Root, tmp_path):
    u.pg.insert('qfc.poi', _poi_rows())

    act = _act(root)
    out = str(tmp_path / 'cli')
    gws.u.ensure_dir(out)

    act.create_package_from_cli(
        'QFC_1',
        out,
        root.app.project('PROJECT_1'),
        root.app.authMgr.systemUser,
    )

    assert os.path.isfile(f'{out}/{packager.COMPLETE_FILE}')
    assert os.path.isfile(f'{out}/qm_qfc_poi.gpkg')
    assert os.path.isfile(f'{out}/QFC_1.qgs')


def test_create_package_from_cli_unknown_project(root: gws.Root, tmp_path):
    act = _act(root)
    with u.raises(gws.NotFoundError):
        act.create_package_from_cli(
            'NO_SUCH_PROJECT',
            str(tmp_path),
            root.app.project('PROJECT_1'),
            root.app.authMgr.systemUser,
        )
