"""Tests for the qfieldcloud patcher."""

from typing import cast

import gws
import gws.test.util as u

from .. import action as action_mod, patcher
from . import util as tu

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
    yield u.gws_root(CONFIG, QGS_PATH=repr(tu.qgs_path('patcher')))


@u.fixture(autouse=True)
def clean():
    u.pg.insert('qfc.poi', [])
    u.pg.insert('qfc.note', [])
    yield


def _args(root, **kwargs):
    act = cast(action_mod.Object, root.get('ACTION_1'))
    qp = act.qfcProjects[0]
    return patcher.Args(
        qfcProject=qp,
        caps=act.get_caps(qp),
        project=root.app.project('PROJECT_1'),
        user=root.app.authMgr.systemUser,
        baseDir='',
        **kwargs,
    )


def _apply(root, *changes) -> bool:
    return patcher.Object().apply_changes(root, _args(root, changes=list(changes)))


def _change(type, layer_uid='poi_L1', **kwargs):
    return patcher.Change(
        uid=kwargs.pop('uid', 'CHANGE_1'),
        type=type,
        layerUid=layer_uid,
        newAtts=kwargs.pop('new', {}),
        oldAtts=kwargs.pop('old', {}),
        wkt=kwargs.pop('wkt', ''),
    )


def _poi():
    return u.pg.rows('SELECT id, name, district_id FROM qfc.poi ORDER BY id')


##


def test_create(root: gws.Root):
    ok = _apply(root, _change('create', new={'id': 1, 'name': 'new poi', 'district_id': 7}))

    assert ok is True
    assert _poi() == [(1, 'new poi', 7)]


def test_create_with_geometry(root: gws.Root):
    _apply(root, _change(
        'create',
        new={'id': 1, 'name': 'geo poi'},
        wkt='POINT(750000 6650000)',
    ))

    rows = u.pg.rows('SELECT ST_X(geom), ST_Y(geom) FROM qfc.poi')
    u.check.close(rows[0], (750000, 6650000), abs_tol=0.001)


def test_create_drops_an_auto_primary_key(root: gws.Root):
    _apply(root, _change('create', 'note_L3', new={'id': 999, 'kind': 'a', 'text': 'auto'}))

    rows = u.pg.rows('SELECT id, text FROM qfc.note')
    assert len(rows) == 1
    assert rows[0][0] != 999
    assert rows[0][1] == 'auto'


def test_create_remaps_fid_gws(root: gws.Root):
    _apply(root, _change('create', 'note_L3', new={'kind': 'a', 'text': 'x', 'fid_gws': 42}))

    assert u.pg.rows('SELECT fid FROM qfc.note') == [(42,)]


def test_update(root: gws.Root):
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'old', 'district_id': 1}])

    ok = _apply(root, _change(
        'patch',
        old={'id': 1, 'name': 'old'},
        new={'name': 'updated'},
    ))

    assert ok is True
    assert _poi() == [(1, 'updated', 1)]


def test_update_geometry(root: gws.Root):
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'x', 'geom': tu.point(1, 1)}])

    _apply(root, _change('patch', old={'id': 1}, new={'name': 'x'}, wkt='POINT(750000 6650000)'))

    rows = u.pg.rows('SELECT ST_X(geom), ST_Y(geom) FROM qfc.poi')
    u.check.close(rows[0], (750000, 6650000), abs_tol=0.001)


def test_delete(root: gws.Root):
    u.pg.insert('qfc.poi', [
        {'id': 1, 'name': 'one'},
        {'id': 2, 'name': 'two'},
    ])

    ok = _apply(root, _change('delete', old={'id': 1}))

    assert ok is True
    assert _poi() == [(2, 'two', None)]


def test_several_changes_in_one_call(root: gws.Root):
    u.pg.insert('qfc.poi', [
        {'id': 1, 'name': 'one'},
        {'id': 2, 'name': 'two'},
    ])

    _apply(
        root,
        _change('create', new={'id': 3, 'name': 'three'}, uid='C1'),
        _change('patch', old={'id': 1}, new={'name': 'ONE'}, uid='C2'),
        _change('delete', old={'id': 2}, uid='C3'),
        _change('create', 'note_L3', new={'kind': 'a', 'text': 'note'}, uid='C4'),
    )

    assert _poi() == [(1, 'ONE', None), (3, 'three', None)]
    assert u.pg.rows('SELECT text FROM qfc.note') == [('note',)]


##


def test_unknown_layer_is_ignored(root: gws.Root):
    ok = _apply(root, _change('create', 'NO_SUCH_LAYER', new={'id': 1, 'name': 'x'}))

    assert ok is False
    assert _poi() == []


def test_non_edit_layer_is_ignored(root: gws.Root):
    ok = _apply(root, _change('create', 'removed_L7', new={'id': 1, 'name': 'x'}))

    assert ok is False
    assert _poi() == []


def test_update_of_a_missing_feature_is_ignored(root: gws.Root):
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'one'}])

    _apply(root, _change('patch', old={'id': 999}, new={'name': 'x'}))

    assert _poi() == [(1, 'one', None)]


def test_delete_of_a_missing_feature_is_ignored(root: gws.Root):
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'one'}])

    _apply(root, _change('delete', old={'id': 999}))

    assert _poi() == [(1, 'one', None)]


##
# uploads


def _upload(root, path, content) -> bool:
    return patcher.Object().apply_upload(root, _args(root, filePath=path, fileContent=content))


def test_upload_is_stored_in_the_content_column(root: gws.Root):
    u.pg.insert('qfc.poi', [
        {'id': 1, 'name': 'one', 'photo': 'DCIM/one.jpg'},
        {'id': 2, 'name': 'two', 'photo': 'DCIM/two.jpg'},
    ])

    ok = _upload(root, 'DCIM/two.jpg', b'IMAGE-BYTES')

    assert ok is True

    rows = u.pg.rows('SELECT id, photo_content FROM qfc.poi ORDER BY id')
    assert [(id, bytes(c) if c else None) for id, c in rows] == [
        (1, None),
        (2, b'IMAGE-BYTES'),
    ]


def test_upload_with_no_matching_feature(root: gws.Root):
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'one', 'photo': 'DCIM/one.jpg'}])

    ok = _upload(root, 'DCIM/other.jpg', b'IMAGE-BYTES')

    assert ok is False
    assert u.pg.rows('SELECT photo_content FROM qfc.poi') == [(None,)]


def test_upload_for_a_model_without_a_file_field(root: gws.Root):
    # qfc.note has an auto-created model with no file field
    ok = _upload(root, 'nothing.jpg', b'IMAGE-BYTES')
    assert ok is False
