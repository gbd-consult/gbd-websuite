"""Tests for ``syncFiles``: file field content sent to QField."""

from typing import cast

import os

import gws
import gws.base.auth.user
import gws.lib.gdalx
import gws.test.util as u

from gws.plugin.qfieldcloud import action as action_mod, packager
from gws.plugin.qfieldcloud._test import util as tu

CONFIG = """
    {DB_PROVIDER}
    projects+ {
        uid "PROJECT_1"
        access "allow all"
        actions+ {
            type "qfieldcloud"
            uid "ACTION_1"
            access "allow all"
            projects+ {
                uid "QFC_1"
                access "allow all"
                provider.path {QGS_PATH}
                {SYNC_FILES}
                models+ {
                    uid "MODEL_POI"
                    type "postgres"
                    dbUid "QFC_DB"
                    tableName "qfc.poi"
                    isEditable true
                    permissions.edit "allow all"
                    fields+ { name "id"    type "integer" }
                    fields+ { name "name"  type "text" }
                    fields+ { name "photo" type "text" }
                    fields+ { name "geom"  type "geometry" }
                    fields+ {
                        type "file"
                        name "photo_file"
                        {FILE_FIELD}
                    }
                }
                {EXTRA_MODELS}
            }
        }
    }
"""

FILE_FIELD = 'contentColumn "photo_content" nameColumn "photo"'


def _root(sync_files='', file_field=FILE_FIELD, extra_models=''):
    return u.gws_root(
        CONFIG,
        QGS_PATH=repr(tu.qgs_path('syncfiles')),
        DB_PROVIDER=tu.db_provider_config(),
        SYNC_FILES=sync_files,
        FILE_FIELD=file_field,
        EXTRA_MODELS=extra_models,
    )


@u.fixture(scope='module', autouse=True)
def tables():
    tu.create_tables()
    yield


def _package(root, out_dir, **args) -> packager.Object:
    act = cast(action_mod.Object, root.get('ACTION_1'))
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
        withData=True,
        withMedia=False,
        withQgis=False,
    )
    pa.update(args)

    po = packager.Object()
    po.create_package(root, pa)
    return po


def _insert_pois(rows):
    u.pg.insert('qfc.poi', [
        {'id': i + 1, 'name': f'poi_{i + 1}', 'photo': photo, 'photo_content': content, 'geom': tu.point(750000 + i, 6650000)}
        for i, (photo, content) in enumerate(rows)
    ])


def _read_poi_gpkg(out_dir):
    with gws.lib.gdalx.open_vector(str(out_dir / 'qm_qfc_poi.gpkg')) as ds:
        la = ds.require_layer('qm_qfc_poi')
        return la.describe(), {r.attributes['id']: r for r in la.get_all()}


def _restricted_user():
    return gws.base.auth.user.AuthorizedUser(None, roles=[gws.c.ROLE_ALL, gws.c.ROLE_USER])


##


def test_config_unknown_field():
    with u.raises(gws.ConfigurationError):
        _root('syncFiles+ { field "no_such_field" }')


def test_config_not_a_file_field():
    with u.raises(gws.ConfigurationError):
        _root('syncFiles+ { field "name" }')


def test_config_ambiguous_field():
    u.pg.create('qfc.sync_extra', {
        'id': 'int primary key',
        'photo': 'text',
        'photo_content': 'bytea',
    })
    extra = """
        models+ {
            uid "MODEL_EXTRA"
            type "postgres"
            dbUid "QFC_DB"
            tableName "qfc.sync_extra"
            fields+ { name "id" type "integer" }
            fields+ { type "file" name "photo_file" contentColumn "photo_content" nameColumn "photo" }
        }
    """
    _root(extra_models=extra)
    with u.raises(gws.ConfigurationError):
        _root('syncFiles+ { field "photo_file" }', extra_models=extra)


def test_config_no_content_column():
    _root(file_field='pathColumn "photo" nameColumn "photo"')
    with u.raises(gws.ConfigurationError):
        _root('syncFiles+ { field "photo_file" }', file_field='pathColumn "photo" nameColumn "photo"')


def test_config_no_name_column():
    with u.raises(gws.ConfigurationError):
        _root('syncFiles+ { field "photo_file" }', file_field='contentColumn "photo_content"')


def test_config_no_name_column_with_blob():
    root = _root('syncFiles+ { field "photo_file" useBlob true }', file_field='contentColumn "photo_content"')
    qp = root.get('ACTION_1').qfcProjects[0]
    assert len(qp.syncFiles) == 1
    assert qp.syncFiles[0].useBlob is True


##


def test_not_listed_fields_are_not_sent(tmp_path):
    _insert_pois([('DCIM/file_1.jpg', b'content_1')])

    po = _package(_root(), tmp_path)

    assert 'DCIM/file_1.jpg' not in po.pathMap
    assert not os.path.exists(tmp_path / 'DCIM/file_1.jpg')
    desc, _ = _read_poi_gpkg(tmp_path)
    assert 'photo_content' not in desc.columnMap


def test_files_are_written(tmp_path):
    _insert_pois([
        ('DCIM/file_1.jpg', b'content_1'),
        ('DCIM/sub/file_2.jpg', b'content_2'),
        ('file_3.jpg', b'content_3'),
    ])

    po = _package(_root('syncFiles+ { field "photo_file" }'), tmp_path)

    for name, content in [('DCIM/file_1.jpg', b'content_1'), ('DCIM/sub/file_2.jpg', b'content_2'), ('file_3.jpg', b'content_3')]:
        path = str(tmp_path / 'cache/files_qm_qfc_poi' / name.replace('/', '_'))
        assert po.pathMap[name] == path
        assert gws.u.read_file_b(path) == content
        assert not os.path.exists(tmp_path / name)

    desc, _ = _read_poi_gpkg(tmp_path)
    assert 'photo_content' not in desc.columnMap


def test_cached_files_are_reused_while_the_size_matches(tmp_path):
    root = _root('syncFiles+ { field "photo_file" }')
    cache_dir = gws.u.ensure_dir(str(tmp_path / 'cache'))
    path = f'{cache_dir}/files_qm_qfc_poi/DCIM_file_1.jpg'

    _insert_pois([('DCIM/file_1.jpg', b'content_1')])
    _package(root, tmp_path / 'package_1', mapCacheDir=cache_dir)
    assert gws.u.read_file_b(path) == b'content_1'

    _insert_pois([('DCIM/file_1.jpg', b'content_2')])
    po = _package(root, tmp_path / 'package_2', mapCacheDir=cache_dir)
    assert po.pathMap['DCIM/file_1.jpg'] == path
    assert gws.u.read_file_b(path) == b'content_1'

    _insert_pois([('DCIM/file_1.jpg', b'content_33')])
    po = _package(root, tmp_path / 'package_3', mapCacheDir=cache_dir)
    assert po.pathMap['DCIM/file_1.jpg'] == path
    assert gws.u.read_file_b(path) == b'content_33'


def test_files_without_name_or_content_are_skipped(tmp_path):
    _insert_pois([
        ('DCIM/file_1.jpg', None),
        ('DCIM/file_2.jpg', b''),
        (None, b'content_3'),
    ])

    po = _package(_root('syncFiles+ { field "photo_file" }'), tmp_path)

    assert sorted(po.pathMap) == ['qm_qfc_district.gpkg', 'qm_qfc_note.gpkg', 'qm_qfc_poi.gpkg']


def test_files_with_invalid_paths_are_skipped(tmp_path):
    _insert_pois([
        ('/tmp/file_1.jpg', b'content_1'),
        ('../file_2.jpg', b'content_2'),
        ('DCIM/../../file_3.jpg', b'content_3'),
    ])

    po = _package(_root('syncFiles+ { field "photo_file" }'), tmp_path)

    assert sorted(po.pathMap) == ['qm_qfc_district.gpkg', 'qm_qfc_note.gpkg', 'qm_qfc_poi.gpkg']
    assert not os.path.exists('/tmp/file_1.jpg')
    assert not os.path.exists(tmp_path.parent / 'file_2.jpg')


def test_files_need_read_permission(tmp_path):
    _insert_pois([('DCIM/file_1.jpg', b'content_1')])

    root = _root('syncFiles+ { field "photo_file" }', file_field=FILE_FIELD + ' permissions.read "allow role_1, deny all"')
    po = _package(root, tmp_path, user=_restricted_user())

    assert 'DCIM/file_1.jpg' not in po.pathMap


##


def test_blobs_are_written(tmp_path):
    _insert_pois([
        ('DCIM/file_1.jpg', b'content_1'),
        ('DCIM/file_2.jpg', None),
        (None, b'content_3'),
    ])

    po = _package(_root('syncFiles+ { field "photo_file" useBlob true }'), tmp_path)

    assert 'DCIM/file_1.jpg' not in po.pathMap

    desc, recs = _read_poi_gpkg(tmp_path)
    assert desc.columnMap['photo_content'].type == gws.AttributeType.bytes
    assert recs[1].attributes['photo_content'] == b'content_1'
    assert not recs[2].attributes['photo_content']
    assert recs[3].attributes['photo_content'] == b'content_3'


def test_blobs_need_read_permission(tmp_path):
    _insert_pois([('DCIM/file_1.jpg', b'content_1')])

    root = _root('syncFiles+ { field "photo_file" useBlob true }', file_field=FILE_FIELD + ' permissions.read "allow role_1, deny all"')
    _package(root, tmp_path, user=_restricted_user())

    desc, _ = _read_poi_gpkg(tmp_path)
    assert 'photo_content' not in desc.columnMap
