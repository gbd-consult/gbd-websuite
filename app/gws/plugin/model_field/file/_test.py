import gws
import gws.lib.image
import gws.lib.mime
import gws.test.util as u

from gws.plugin.model_field.file import FileValue


def _model_cfg(uid, extra='', field_extra=''):
    return f"""
        models+ {{
            uid {uid!r}
            type "postgres"
            tableName "files"
            isEditable true
            {extra}
            fields+ {{ name "id" type "integer" isPrimaryKey true }}
            fields+ {{ name "filename" type "text" }}
            fields+ {{ name "file" type "file" contentColumn "content" nameColumn "filename" {field_extra} }}
        }}
    """


@u.fixture(scope='module')
def root():
    u.pg.create('files', {'id': 'int primary key', 'filename': 'text', 'content': 'bytea'})
    u.pg.insert(
        'files',
        [
            {'id': 1, 'filename': 'a.txt', 'content': b'AAA'},
            {'id': 2, 'filename': 'b.txt', 'content': b'BBB'},
            {'id': 4, 'filename': 'd.png', 'content': gws.lib.image.from_size((300, 200)).to_bytes(gws.lib.mime.PNG)},
            {'id': 5, 'filename': 'e.png', 'content': b'NOT_AN_IMAGE'},
        ],
    )

    cfg = f"""
        permissions.all "allow all"

        actions+ {{ type "web" }}
        actions+ {{ type "edit" }}

        projects+ {{
            uid "A"
            {_model_cfg('OPEN')} 
            {_model_cfg('FIELD_DENIED', field_extra='permissions.read "deny all"')}
            {_model_cfg('MODEL_DENIED', extra='permissions.read "deny all"')}
            {_model_cfg('FILTERED', extra='sqlFilter "id = 1"')}
        }}
    """

    yield u.gws_root(cfg)


def _url(model_uid, feature_uid, **kwargs):
    return gws.u.action_url_path(
        'webFile',
        projectUid='A',
        modelUid=model_uid,
        fieldName='file',
        featureUid=feature_uid,
        **kwargs,
    )


##


def test_download(root: gws.Root):
    res = u.http.get(root, _url('OPEN', '1'))

    assert res.status_code == 200
    assert res.data == b'AAA'
    assert res.mimetype == 'text/plain'
    assert res.headers['Content-Disposition'] == 'attachment; filename="a.txt"'


def test_download_preview_returns_a_thumbnail(root: gws.Root):
    res = u.http.get(root, _url('OPEN', '4', preview=1))

    assert res.status_code == 200
    assert res.mimetype == 'image/png'
    assert 'Content-Disposition' not in res.headers
    assert gws.lib.image.from_bytes(res.data).size() == (120, 80)


def test_download_preview_is_refused_for_non_images(root: gws.Root):
    assert u.http.get(root, _url('OPEN', '1', preview=1)).status_code == 404


def test_download_preview_is_refused_for_undecodable_images(root: gws.Root):
    assert u.http.get(root, _url('OPEN', '5')).status_code == 200
    assert u.http.get(root, _url('OPEN', '5', preview=1)).status_code == 404


def test_download_unknown_feature(root: gws.Root):
    assert u.http.get(root, _url('OPEN', '999')).status_code == 404


def test_download_unknown_field(root: gws.Root):
    url = gws.u.action_url_path(
        'webFile',
        projectUid='A',
        modelUid='OPEN',
        fieldName='NOT_A_FIELD',
        featureUid='1',
    )
    assert u.http.get(root, url).status_code == 404


def test_download_denied_for_unreadable_field(root: gws.Root):
    assert u.http.get(root, _url('OPEN', '1')).status_code == 200
    assert u.http.get(root, _url('FIELD_DENIED', '1')).status_code == 404


def test_download_denied_for_unreadable_model(root: gws.Root):
    assert u.http.get(root, _url('MODEL_DENIED', '1')).status_code == 403


def test_download_respects_the_model_filter(root: gws.Root):
    assert u.http.get(root, _url('OPEN', '2')).status_code == 200

    assert u.http.get(root, _url('FILTERED', '1')).status_code == 200
    assert u.http.get(root, _url('FILTERED', '2')).status_code == 404


def test_props_contain_the_download_url(root: gws.Root):
    res = u.http.api(root, 'editGetFeature', dict(projectUid='A', modelUid='OPEN', featureUid='1'))

    fp = res.json['feature']['attributes']['file']
    assert fp['label'] == 'a.txt'
    assert fp['size'] == 3
    assert fp['downloadUrl'] == _url('OPEN', '1') + '/a.txt'
    assert fp['previewUrl'] == ''


def test_props_contain_the_preview_url_for_images(root: gws.Root):
    res = u.http.api(root, 'editGetFeature', dict(projectUid='A', modelUid='OPEN', featureUid='4'))

    fp = res.json['feature']['attributes']['file']
    assert fp['previewUrl'] == gws.u.action_url_path(
        'webFile',
        preview=1,
        projectUid='A',
        modelUid='OPEN',
        fieldName='file',
        featureUid='4',
    ) + '/d.png'


def test_props_omit_an_unreadable_field(root: gws.Root):
    res = u.http.api(root, 'editGetFeature', dict(projectUid='A', modelUid='FIELD_DENIED', featureUid='1'))

    assert 'filename' in res.json['feature']['attributes']
    assert 'file' not in res.json['feature']['attributes']


def test_create_stores_the_content(root: gws.Root):
    mm = u.cast(gws.Model, root.get('OPEN'))

    f = u.model.feature(mm, id=3, file=FileValue(content=b'CCC', name='c.txt'))
    mm.create_feature(f, u.model.context(op=gws.ModelOperation.create))

    rows = u.pg.rows('SELECT id, filename, content FROM files WHERE id=3')
    assert [(r[0], r[1], bytes(r[2])) for r in rows] == [
        (3, 'c.txt', b'CCC'),
    ]

    res = u.http.get(root, _url('OPEN', '3'))
    assert res.status_code == 200
    assert res.data == b'CCC'
