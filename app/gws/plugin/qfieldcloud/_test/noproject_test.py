"""Tests for an application-level action, without a GWS project."""

from typing import cast

import io

import gws
import gws.lib.jsonx
import gws.test.util as u

from gws.plugin.qfieldcloud import action as action_mod, core
from gws.plugin.qfieldcloud._test import util as tu

CONFIG = f"""
    auth.providers+ {{ type "{u.auth.PROVIDER_1}" }}
    auth.session {{ type "sqlite" }}

    actions+ {{
        type "qfieldcloud"
        uid "ACTION_1"
        access "allow all"
        projects+ {{
            uid "QFC_1"
            access "allow all"
            provider.path {{QGS_PATH}}
        }}
    }}

    projects+ {{
        uid "PROJECT_1"
        access "allow all"
    }}
"""

ENDPOINT = '/_/qfieldcloudApi'


@u.fixture(scope='module')
def root():
    tu.create_tables()

    def patch(root_el):
        tu.set_project_prop(root_el, 'createBaseMap', '0', 'int')
        tu.remove_media_dirs(root_el)

    u.auth.add_user('user1', 'pass1')

    yield u.gws_root(CONFIG, QGS_PATH=repr(tu.qgs_path('noproject', patch)))


def _act(root) -> action_mod.Object:
    return cast(action_mod.Object, root.get('ACTION_1'))


def _qfc_project(root) -> core.QfcProject:
    return _act(root).qfcProjects[0]


def _url(path):
    return f'{ENDPOINT}/{path}'


def _auth(token):
    return {'Authorization': f'Token {token}'}


@u.fixture
def token(root):
    res = u.http.post(root, _url('api/v1/auth/token'), json={'username': 'user1', 'password': 'pass1'})
    assert res.status_code == 200
    return res.json['token']


##


def test_the_action_has_no_project(root: gws.Root):
    assert _act(root).find_closest(gws.ext.object.project) is None


def test_the_project_is_listed(root: gws.Root, token):
    res = u.http.get(root, _url('api/v1/projects'), headers=_auth(token))

    assert res.status_code == 200
    assert [p['id'] for p in res.json] == ['QFC_1']


def test_a_project_uid_prefix_is_accepted(root: gws.Root, token):
    res = u.http.get(root, f'{ENDPOINT}/projectUid/PROJECT_1/api/v1/projects', headers=_auth(token))

    assert res.status_code == 200
    assert [p['id'] for p in res.json] == ['QFC_1']


def test_an_unknown_project_uid_prefix_is_refused(root: gws.Root, token):
    res = u.http.get(root, f'{ENDPOINT}/projectUid/NO_SUCH_PROJECT/api/v1/projects', headers=_auth(token))

    assert res.status_code == 404


def test_packaging_over_the_api(root: gws.Root, token):
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'one', 'geom': tu.point(750000, 6650000)}])

    res = u.http.post(root, _url('api/v1/jobs'), json={'project_id': 'QFC_1', 'type': 'package'}, headers=_auth(token))
    assert res.status_code == 200
    assert res.json['status'] == 'finished'

    res = u.http.get(root, _url('api/v1/packages/QFC_1/latest'), headers=_auth(token))

    assert res.status_code == 200
    assert sorted(f['name'] for f in res.json['files']) == [
        'QFC_1.qgs', 'qm_qfc_district.gpkg', 'qm_qfc_note.gpkg', 'qm_qfc_poi.gpkg',
    ]


def _job_payload(root, url, token):
    res = u.http.post(root, url, json={'project_id': 'QFC_1', 'type': 'package'}, headers=_auth(token))
    assert res.status_code == 200
    job = root.app.jobMgr.get_job(res.json['id'])
    return action_mod.WorkerPayload(gws.u.require(job).payload)


def test_a_job_without_project_uid_has_no_project(root: gws.Root, token):
    pa = _job_payload(root, _url('api/v1/jobs'), token)
    assert pa.projectUid is None


def test_a_job_with_a_project_uid_prefix_uses_the_project(root: gws.Root, token):
    pa = _job_payload(root, f'{ENDPOINT}/projectUid/PROJECT_1/api/v1/jobs', token)
    assert pa.projectUid == 'PROJECT_1'


def test_deltas_over_the_api(root: gws.Root, token):
    u.pg.insert('qfc.poi', [])

    payload = {
        'id': 'PAYLOAD_1',
        'project': 'QFC_1',
        'version': '1.0',
        'files': [],
        'deltas': [
            {
                'uuid': 'DELTA_1',
                'clientId': 'CLIENT_1',
                'localLayerId': 'poi_L1',
                'method': 'create',
                'new': {'attributes': {'id': 1, 'name': 'created'}},
                'old': None,
            },
        ],
    }

    data = {'file': (io.BytesIO(gws.lib.jsonx.to_string(payload).encode('utf8')), 'deltafile.json')}
    res = u.http.post(root, _url('api/v1/deltas/QFC_1'), data=data, headers=_auth(token))

    assert res.status_code == 200
    assert u.pg.rows('SELECT id, name FROM qfc.poi') == [(1, 'created')]


def test_packaging_from_the_cli(root: gws.Root, tmp_path):
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'one', 'geom': tu.point(750000, 6650000)}])

    _act(root).create_package_from_cli('QFC_1', str(tmp_path), None, root.app.authMgr.systemUser)

    assert (tmp_path / 'qm_qfc_poi.gpkg').exists()
