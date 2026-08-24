"""Tests for the qfieldcloud HTTP API."""

from typing import cast

import io
import os

import gws
import gws.lib.jsonx
import gws.test.util as u

from gws.plugin.qfieldcloud import action as action_mod, packager
from gws.plugin.qfieldcloud._test import util as tu

CONFIG = f"""
    auth.providers+ {{ type "{u.auth.PROVIDER_1}" }}
    auth.session {{ type "sqlite" }}
    auth.methods+ {{ type web secure False cookieName AUTH_COOKIE }}

    actions+ {{ type auth access "allow all" }}

    projects+ {{
        uid "PROJECT_1"
        access "allow all"
        actions+ {{
            type "qfieldcloud"
            uid "ACTION_1"
            access "allow all"
            auth.secure False
            projects+ {{
                uid "QFC_1"
                title "QField Test"
                access "allow all"
                provider.path {{QGS_PATH}}
                models+ {{
                    uid "MODEL_POI"
                    type "postgres"
                    tableName "qfc.poi"
                    isEditable true
                    permissions.edit "allow all"
                    fields+ {{ name "id"          type "integer" }}
                    fields+ {{ name "name"        type "text" }}
                    fields+ {{ name "district_id" type "integer" }}
                    fields+ {{ name "geom"        type "geometry" }}
                    fields+ {{
                        type "file"
                        name "photo_file"
                        contentColumn "photo_content"
                        nameColumn "photo"
                    }}
                }}
            }}
            projects+ {{
                uid "QFC_SECRET"
                title "Secret"
                access "allow role1, deny all"
                provider.path {{QGS_PATH}}
            }}
        }}
    }}

    projects+ {{
        uid "PROJECT_2"
        access "allow all"
    }}
"""

ENDPOINT = '/_/qfieldcloudApi'


def _url(path, project_uid='PROJECT_1'):
    return f'{ENDPOINT}/projectUid/{project_uid}/{path}'


def _auth(token):
    return {'Authorization': f'Token {token}'}


@u.fixture(scope='module')
def root():
    tu.create_tables()

    def patch(root_el):
        # keep the api tests independent of the qgis server
        tu.set_project_prop(root_el, 'createBaseMap', '0', 'int')
        tu.remove_media_dirs(root_el)

    u.auth.add_user('user1', 'pass1', displayName='User One', roles=['role1'])
    u.auth.add_user('user2', 'pass2', displayName='User Two')

    yield u.gws_root(CONFIG, QGS_PATH=repr(tu.qgs_path('api', patch)))


def _token(root, username='user1', password='pass1'):
    res = u.http.post(root, _url('api/v1/auth/token'), json={'username': username, 'password': password})
    assert res.status_code == 200
    return res.json['token']


@u.fixture
def token(root):
    return _token(root)


##
# routing


def test_unknown_route(root: gws.Root, token):
    assert u.http.get(root, _url('api/v1/nope'), headers=_auth(token)).status_code == 404


def test_empty_path(root: gws.Root):
    assert u.http.get(root, ENDPOINT + '/').status_code == 404


def test_project_uid_mismatch(root: gws.Root, token):
    # the action lives in PROJECT_1, PROJECT_2 has no qfieldcloud action
    res = u.http.get(root, _url('api/v1/projects', 'PROJECT_2'), headers=_auth(token))
    assert res.status_code == 404


def test_unknown_gws_project(root: gws.Root, token):
    res = u.http.get(root, _url('api/v1/projects', 'NO_SUCH_PROJECT'), headers=_auth(token))
    assert res.status_code == 404


##
# authorisation


def test_auth_providers_is_public(root: gws.Root):
    res = u.http.get(root, _url('api/v1/auth/providers'))

    assert res.status_code == 200
    assert res.json == [{'type': 'credentials', 'id': 'credentials', 'name': 'Username / Password'}]


def test_auth_token(root: gws.Root):
    res = u.http.post(root, _url('api/v1/auth/token'), json={'username': 'user1', 'password': 'pass1'})

    assert res.status_code == 200
    assert res.json['username'] == 'user1'
    assert res.json['full_name'] == 'User One'
    assert res.json['token']
    assert res.json['expires_at']


def test_auth_token_wrong_credentials(root: gws.Root):
    assert u.http.post(root, _url('api/v1/auth/token'), json={'username': 'user1', 'password': 'XX'}).status_code == 403
    assert u.http.post(root, _url('api/v1/auth/token'), json={'username': 'XX', 'password': 'pass1'}).status_code == 403
    assert u.http.post(root, _url('api/v1/auth/token'), json={}).status_code == 403


def test_protected_route_without_a_token(root: gws.Root):
    assert u.http.get(root, _url('api/v1/auth/user')).status_code == 403


def test_protected_route_with_a_malformed_header(root: gws.Root):
    res = u.http.get(root, _url('api/v1/auth/user'), headers={'Authorization': 'Bearer xyz'})
    assert res.status_code == 403


def test_protected_route_with_a_web_session_cookie(root: gws.Root):
    # a session created by the web method must not be usable as a qfieldcloud token
    res = u.http.api(root, 'authLogin', {'username': 'user1', 'password': 'pass1'})
    assert res.status_code == 200

    sid = res.cookies['AUTH_COOKIE'].value
    assert u.http.get(root, _url('api/v1/auth/user'), headers=_auth(sid)).status_code == 403


def test_protected_route_with_an_invalid_token(root: gws.Root):
    assert u.http.get(root, _url('api/v1/auth/user'), headers=_auth('NOT-A-TOKEN')).status_code == 403


def test_auth_user(root: gws.Root, token):
    res = u.http.get(root, _url('api/v1/auth/user'), headers=_auth(token))

    assert res.status_code == 200
    assert res.json['username'] == 'user1'
    assert res.json['full_name'] == 'User One'


def test_logout_invalidates_the_token(root: gws.Root):
    tok = _token(root)
    assert u.http.get(root, _url('api/v1/auth/user'), headers=_auth(tok)).status_code == 200

    assert u.http.post(root, _url('api/v1/auth/logout'), headers=_auth(tok)).status_code == 200
    assert u.http.get(root, _url('api/v1/auth/user'), headers=_auth(tok)).status_code == 403


##
# projects


def test_projects_are_filtered_by_access(root: gws.Root):
    res = u.http.get(root, _url('api/v1/projects'), headers=_auth(_token(root, 'user1', 'pass1')))
    assert sorted(p['id'] for p in res.json) == ['QFC_1', 'QFC_SECRET']

    res = u.http.get(root, _url('api/v1/projects'), headers=_auth(_token(root, 'user2', 'pass2')))
    assert [p['id'] for p in res.json] == ['QFC_1']


def test_projects_limit_and_offset(root: gws.Root, token):
    res = u.http.get(root, _url('api/v1/projects'), headers=_auth(token), query_string={'limit': 1})
    assert [p['id'] for p in res.json] == ['QFC_1']

    res = u.http.get(root, _url('api/v1/projects'), headers=_auth(token), query_string={'limit': 1, 'offset': 1})
    assert [p['id'] for p in res.json] == ['QFC_SECRET']


def test_project_by_id(root: gws.Root, token):
    res = u.http.get(root, _url('api/v1/projects/QFC_1'), headers=_auth(token))

    assert res.status_code == 200
    assert res.json['id'] == 'QFC_1'
    assert res.json['name'] == 'QField Test'
    assert res.json['owner'] == 'user1'


def test_unknown_project_by_id(root: gws.Root, token):
    assert u.http.get(root, _url('api/v1/projects/NOPE'), headers=_auth(token)).status_code == 404


def test_forbidden_project_by_id(root: gws.Root):
    tok = _token(root, 'user2', 'pass2')
    assert u.http.get(root, _url('api/v1/projects/QFC_SECRET'), headers=_auth(tok)).status_code == 404


##
# jobs and packages


def _package(root, token):
    u.pg.insert('qfc.poi', [
        {'id': 1, 'name': 'one', 'photo': 'DCIM/one.jpg', 'geom': tu.point(750000, 6650000)},
    ])
    res = u.http.post(
        root,
        _url('api/v1/jobs'),
        json={'project_id': 'QFC_1', 'type': 'package'},
        headers=_auth(token),
    )
    assert res.status_code == 200
    return res.json


def test_create_package_job(root: gws.Root, token):
    job = _package(root, token)

    assert job['status'] == 'finished'
    assert job['type'] == 'package'
    assert job['project_id'] == 'QFC_1'


def test_unsupported_job_type(root: gws.Root, token):
    res = u.http.post(
        root,
        _url('api/v1/jobs'),
        json={'project_id': 'QFC_1', 'type': 'delta_apply'},
        headers=_auth(token),
    )
    assert res.status_code >= 400


def test_job_by_id(root: gws.Root, token):
    job = _package(root, token)

    res = u.http.get(root, _url('api/v1/jobs/' + job['id']), headers=_auth(token))
    assert res.status_code == 200
    assert res.json['id'] == job['id']


def test_job_of_another_user(root: gws.Root, token):
    job = _package(root, token)

    tok2 = _token(root, 'user2', 'pass2')
    assert u.http.get(root, _url('api/v1/jobs/' + job['id']), headers=_auth(tok2)).status_code == 404


def test_unknown_job(root: gws.Root, token):
    assert u.http.get(root, _url('api/v1/jobs/NOPE'), headers=_auth(token)).status_code == 404


def test_package_listing(root: gws.Root, token):
    _package(root, token)

    res = u.http.get(root, _url('api/v1/packages/QFC_1/latest'), headers=_auth(token))

    assert res.status_code == 200
    assert res.json['status'] == 'finished'
    assert res.json['package_id'] == 'QFC_1'

    names = sorted(f['name'] for f in res.json['files'])
    assert names == ['QFC_1.qgs', 'qm_qfc_district.gpkg', 'qm_qfc_note.gpkg', 'qm_qfc_poi.gpkg']

    for f in res.json['files']:
        assert f['size'] > 0
        assert len(f['sha256']) == 64
        assert f['is_attachment'] is False


def test_files_listing(root: gws.Root, token):
    _package(root, token)

    res = u.http.get(root, _url('api/v1/files/QFC_1'), headers=_auth(token))

    assert res.status_code == 200
    assert sorted(f['name'] for f in res.json) == ['QFC_1.qgs', 'qm_qfc_district.gpkg', 'qm_qfc_note.gpkg', 'qm_qfc_poi.gpkg']


def test_package_file_download(root: gws.Root, token):
    _package(root, token)

    res = u.http.get(root, _url('api/v1/packages/QFC_1/latest/files/QFC_1.qgs'), headers=_auth(token))

    assert res.status_code == 200
    assert res.get_data().startswith(b'<qgis')


def test_unknown_package_file(root: gws.Root, token):
    _package(root, token)

    res = u.http.get(root, _url('api/v1/packages/QFC_1/latest/files/nope.gpkg'), headers=_auth(token))
    assert res.status_code == 404


##
# deltas


def _delta_payload(payload_id, deltas):
    return {
        'id': payload_id,
        'project': 'QFC_1',
        'version': '1.0',
        'files': [],
        'deltas': deltas,
    }


def _delta(method, uid='DELTA_1', layer='poi_L1', new=None, old=None, geometry=None):
    d = {
        'uuid': uid,
        'clientId': 'CLIENT_1',
        'localLayerId': layer,
        'method': method,
        'new': None,
        'old': None,
    }
    if new is not None:
        d['new'] = {'attributes': new}
        if geometry:
            d['new']['geometry'] = geometry
    if old is not None:
        d['old'] = {'attributes': old}
    return d


def _post_deltas(root, token, payload, boundary=None):
    data = {'file': (io.BytesIO(gws.lib.jsonx.to_string(payload).encode('utf8')), 'deltafile.json')}
    kwargs = {}
    if boundary:
        kwargs['content_type'] = f'multipart/form-data; boundary={boundary}'
    return u.http.post(root, _url('api/v1/deltas/QFC_1'), data=data, headers=_auth(token), **kwargs)


def test_post_deltas(root: gws.Root, token):
    u.pg.insert('qfc.poi', [])

    res = _post_deltas(root, token, _delta_payload('PAYLOAD_1', [
        _delta('create', 'D1', new={'id': 1, 'name': 'created'}, geometry='POINT(750000 6650000)'),
    ]))

    assert res.status_code == 200
    assert u.pg.rows('SELECT id, name FROM qfc.poi') == [(1, 'created')]


def test_post_deltas_stores_the_payload(root: gws.Root, token):
    u.pg.insert('qfc.poi', [])

    _post_deltas(root, token, _delta_payload('PAYLOAD_2', [
        _delta('create', 'D2', new={'id': 2, 'name': 'x'}),
    ]))

    res = u.http.get(root, _url('api/v1/deltas/QFC_1/PAYLOAD_2'), headers=_auth(token))

    assert res.status_code == 200
    assert len(res.json) == 1
    assert res.json[0]['id'] == 'D2'
    assert res.json[0]['deltafile_id'] == 'PAYLOAD_2'
    assert res.json[0]['status'] == 'STATUS_APPLIED'
    assert res.json[0]['last_status'] == 'applied'
    assert res.json[0]['created_by'] == 'user1'


def test_post_deltas_with_a_qt_boundary(root: gws.Root, token):
    # Qt boundaries contain base64 characters that werkzeug's header parser does not understand
    u.pg.insert('qfc.poi', [])

    res = _post_deltas(
        root,
        token,
        _delta_payload('PAYLOAD_QT', [_delta('create', 'DQ', new={'id': 5, 'name': 'qt'})]),
        boundary='boundary_.oOo._MTIzNDU2Nzg5/+ab==',
    )

    assert res.status_code == 200
    assert u.pg.rows('SELECT id, name FROM qfc.poi') == [(5, 'qt')]


def test_post_deltas_with_invalid_content(root: gws.Root, token):
    data = {'file': (io.BytesIO(b'not json'), 'deltafile.json')}
    res = u.http.post(root, _url('api/v1/deltas/QFC_1'), data=data, headers=_auth(token))
    assert res.status_code == 400


def test_post_deltas_without_a_file(root: gws.Root, token):
    res = u.http.post(root, _url('api/v1/deltas/QFC_1'), data={'x': 'y'}, headers=_auth(token))
    assert res.status_code == 400


def test_unknown_delta_payload(root: gws.Root, token):
    assert u.http.get(root, _url('api/v1/deltas/QFC_1/NOPE'), headers=_auth(token)).status_code == 404


def test_delta_payload_of_another_user(root: gws.Root, token):
    u.pg.insert('qfc.poi', [])
    _post_deltas(root, token, _delta_payload('PAYLOAD_3', [
        _delta('create', 'D3', new={'id': 3, 'name': 'x'}),
    ]))

    tok2 = _token(root, 'user2', 'pass2')
    res = u.http.get(root, _url('api/v1/deltas/QFC_1/PAYLOAD_3'), headers=_auth(tok2))
    assert res.status_code == 404


##
# file uploads


def test_post_file(root: gws.Root, token):
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'one', 'photo': 'DCIM/one.jpg'}])

    data = {'file': (io.BytesIO(b'JPEG-BYTES'), 'one.jpg')}
    res = u.http.post(root, _url('api/v1/files/QFC_1/DCIM/one.jpg'), data=data, headers=_auth(token))

    assert res.status_code == 200

    rows = u.pg.rows('SELECT photo_content FROM qfc.poi')
    assert bytes(rows[0][0]) == b'JPEG-BYTES'


def test_post_file_without_a_file(root: gws.Root, token):
    res = u.http.post(root, _url('api/v1/files/QFC_1/DCIM/one.jpg'), data={'x': 'y'}, headers=_auth(token))
    assert res.status_code == 400


##
# housekeeping


def test_latest_package_dir_ignores_incomplete_packages(root: gws.Root):
    act = cast(action_mod.Object, root.get('ACTION_1'))
    qp = act.qfcProjects[0]

    base = act.fs_project_base_dir(qp)
    for d in os.listdir(base):
        if d.startswith('package_'):
            gws.lib.osx.rmdir(f'{base}/{d}')

    good = act.fs_new_package_dir(qp, '20260101000000001')
    gws.u.write_file(f'{good}/{packager.COMPLETE_FILE}', '1')
    act.fs_new_package_dir(qp, '20260101000000002')

    assert act.fs_latest_package_dir(qp) == good


def test_cleanup_old_packages(root: gws.Root):
    act = cast(action_mod.Object, root.get('ACTION_1'))
    qp = act.qfcProjects[0]

    old = act.fs_new_package_dir(qp, '20260101000000003')
    os.utime(old, (1000, 1000))
    new = act.fs_new_package_dir(qp, '20260101000000004')

    act.fs_cleanup_old_packages(qp, keep_seconds=3600)

    assert not os.path.isdir(old)
    assert os.path.isdir(new)


def test_cleanup_old_deltas(root: gws.Root):
    act = cast(action_mod.Object, root.get('ACTION_1'))
    qp = act.qfcProjects[0]

    old = act.fs_delta_payload_path(qp, 'OLD_PAYLOAD')
    gws.u.write_file(old, '[]')
    os.utime(old, (1000, 1000))

    new = act.fs_delta_payload_path(qp, 'NEW_PAYLOAD')
    gws.u.write_file(new, '[]')

    act.fs_cleanup_old_deltas(qp, keep_seconds=3600)

    assert not os.path.isfile(old)
    assert os.path.isfile(new)
