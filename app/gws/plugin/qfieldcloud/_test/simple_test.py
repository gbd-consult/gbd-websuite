"""Tests for the minimal configuration, without models."""

from typing import cast

import io

import gws
import gws.lib.gdalx
import gws.lib.jsonx
import gws.test.util as u

from gws.plugin.qfieldcloud import action as action_mod, caps, core, packager
from gws.plugin.qfieldcloud._test import util as tu

CONFIG = f"""
    auth.providers+ {{ type "{u.auth.PROVIDER_1}" }}
    auth.session {{ type "sqlite" }}

    projects+ {{
        uid "PROJECT_1"
        access "allow all"
        actions+ {{
            type "qfieldcloud"
            uid "ACTION_1"
            projects+ {{
                provider.path {{QGS_PATH}}
            }}
        }}
    }}
"""

ENDPOINT = '/_/qfieldcloudApi'


@u.fixture(scope='module')
def root():
    tu.create_tables()

    def patch(root_el):
        # keep the tests independent of the qgis server
        tu.set_project_prop(root_el, 'createBaseMap', '0', 'int')
        tu.remove_media_dirs(root_el)

    u.auth.add_user('user1', 'pass1')

    yield u.gws_root(CONFIG, QGS_PATH=repr(tu.qgs_path('simple', patch)))


def _act(root) -> action_mod.Object:
    return cast(action_mod.Object, root.get('ACTION_1'))


def _qfc_project(root) -> core.QfcProject:
    return _act(root).qfcProjects[0]


def _caps(root) -> caps.Caps:
    return _act(root).get_caps(_qfc_project(root))


def _url(path):
    return f'{ENDPOINT}/projectUid/PROJECT_1/{path}'


def _auth(token):
    return {'Authorization': f'Token {token}'}


@u.fixture
def token(root):
    res = u.http.post(root, _url('api/v1/auth/token'), json={'username': 'user1', 'password': 'pass1'})
    assert res.status_code == 200
    return res.json['token']


##
# configuration


def test_a_single_project_is_configured(root: gws.Root):
    assert len(_act(root).qfcProjects) == 1


def test_the_title_defaults_to_the_uid(root: gws.Root):
    qp = _qfc_project(root)
    assert qp.uid
    assert qp.title == qp.uid


def test_no_models_are_configured(root: gws.Root):
    assert _qfc_project(root).models == []


##
# generated models


def test_models_are_created_for_offline_layers(root: gws.Root):
    cs = _caps(root)
    assert sorted(cs.modelMap) == ['qm_qfc_district', 'qm_qfc_note', 'qm_qfc_poi']


def test_a_created_model_is_editable(root: gws.Root):
    me = _caps(root).modelMap['qm_qfc_poi']

    assert me.tableName == 'qfc.poi'
    assert me.model.isEditable is True


def test_a_created_model_contains_all_columns(root: gws.Root):
    me = _caps(root).modelMap['qm_qfc_poi']

    # there is no field type for the "bytea" column "photo_content"
    assert sorted(f.name for f in me.model.fields) == ['district_id', 'geom', 'id', 'name', 'photo']


def test_offline_layers_use_the_created_models(root: gws.Root):
    cs = _caps(root)

    le = cs.layerMap['poi_L1']
    assert le.action == caps.LayerAction.edit
    assert le.modelEntry is cs.modelMap['qm_qfc_poi']
    assert le.dataSource == './qm_qfc_poi.gpkg|layername=qm_qfc_poi'

    # a fully locked layer is exported as well
    assert cs.layerMap['district_L2'].action == caps.LayerAction.edit
    assert cs.layerMap['district_L2'].readOnly is True


##
# packaging


def test_the_package_contains_the_data(root: gws.Root, tmp_path):
    u.pg.insert('qfc.poi', [
        {'id': 1, 'name': 'one', 'geom': tu.point(750000, 6650000)},
        {'id': 2, 'name': 'two', 'geom': tu.point(751000, 6651000)},
    ])

    act = _act(root)
    qp = _qfc_project(root)

    packager.Object().create_package(root, packager.Args(
        uid='TEST',
        qfcProject=qp,
        caps=act.get_caps(qp),
        project=root.app.project('PROJECT_1'),
        user=root.app.authMgr.systemUser,
        packageDir=str(tmp_path),
        mapCacheDir=gws.u.ensure_dir(str(tmp_path) + '/cache'),
        withBaseMap=False,
        withData=True,
        withMedia=False,
        withQgis=True,
    ))

    with gws.lib.gdalx.open_vector(str(tmp_path / 'qm_qfc_poi.gpkg')) as ds:
        la = ds.require_layer('qm_qfc_poi')
        recs = la.get_all()
        desc = la.describe()

    assert sorted(r.attributes['name'] for r in recs) == ['one', 'two']
    # bytea has no gpkg counterpart
    assert 'photo_content' not in desc.columnMap
    assert 'photo' in desc.columnMap


##
# the API


def test_the_project_is_listed(root: gws.Root, token):
    qp = _qfc_project(root)

    res = u.http.get(root, _url('api/v1/projects'), headers=_auth(token))

    assert res.status_code == 200
    assert [p['id'] for p in res.json] == [qp.uid]
    assert res.json[0]['name'] == qp.uid


def test_packaging_over_the_api(root: gws.Root, token):
    qp = _qfc_project(root)
    u.pg.insert('qfc.poi', [{'id': 1, 'name': 'one', 'geom': tu.point(750000, 6650000)}])

    res = u.http.post(
        root,
        _url('api/v1/jobs'),
        json={'project_id': qp.uid, 'type': 'package'},
        headers=_auth(token),
    )
    assert res.status_code == 200
    assert res.json['status'] == 'finished'

    res = u.http.get(root, _url(f'api/v1/packages/{qp.uid}/latest'), headers=_auth(token))

    assert res.status_code == 200
    assert sorted(f['name'] for f in res.json['files']) == [
        f'{qp.uid}.qgs', 'qm_qfc_district.gpkg', 'qm_qfc_note.gpkg', 'qm_qfc_poi.gpkg',
    ]


def test_deltas_over_the_api(root: gws.Root, token):
    qp = _qfc_project(root)
    u.pg.insert('qfc.poi', [])

    payload = {
        'id': 'PAYLOAD_1',
        'project': qp.uid,
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
    res = u.http.post(root, _url(f'api/v1/deltas/{qp.uid}'), data=data, headers=_auth(token))

    assert res.status_code == 200
    assert u.pg.rows('SELECT id, name FROM qfc.poi') == [(1, 'created')]
