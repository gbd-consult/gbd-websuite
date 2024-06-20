import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('plain', {'id': 'int primary key', 'name': 'text', 'g': 'geometry(point,3857)'})
    u.pg.insert('plain', [
        dict(id=1, name='a11', g=u.ewkb('POINT(10 100)')),
        dict(id=2, name='a22', g=u.ewkb('POINT(20 200)')),
        dict(id=3, name='a33', g=u.ewkb('POINT(30 300)')),
    ])

    cfg = '''
        permissions.all "allow all"

        models+ {
            uid "PLAIN" type "postgres" tableName "plain"
        }
        actions+ {
            type 'edit'
        }
        projects+ {
            uid "A"
            templates+ {
                subject "feature.title"
                type "html"
                text "--{id}/{name}--"
            }
            models+ {
                uid "MODEL_PLAIN"
                type "postgres"
                tableName "plain"
                isEditable true
                fields+ { name "id" type "integer" isPrimaryKey true }
                fields+ { name "name" type "text" }
                fields+ { name "g" type "geometry" }
            }
        }
    '''

    yield u.gws_root(cfg)


def test_get_models(root: gws.Root):
    res = u.api_request(root, 'editGetModels', dict(projectUid='A'))
    ms = res.json['models']
    assert ms[0]['uid'] == 'MODEL_PLAIN'
    assert len(ms[0]['fields']) == 3


def test_get_feature(root: gws.Root):
    res = u.api_request(root, 'editGetFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', featureUid='2'))
    props = res.json['feature']
    assert props['attributes']['id'] == 2
    assert props['attributes']['name'] == 'a22'


def test_write_feature(root: gws.Root):
    res = u.api_request(root, 'editGetFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', featureUid='2'))
    props = res.json['feature']
    props['attributes']['name'] = 'a22-new'

    res = u.api_request(root, 'editWriteFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', feature=props))
    assert res.status_code == 200

    assert u.pg.rows('SELECT id,name FROM plain ORDER BY id') == [
        (1, 'a11'),
        (2, 'a22-new'),
        (3, 'a33'),
    ]


def test_create_feature(root: gws.Root):
    res = u.api_request(root, 'editGetFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', featureUid='2'))
    props = res.json['feature']
    props['attributes'] = {'id': 777, 'name': 'NEW_NAME'}
    props['isNew'] = True

    res = u.api_request(root, 'editWriteFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', feature=props))
    assert res.status_code == 200

    assert u.pg.rows('SELECT id,name FROM plain WHERE id=777') == [
        (777, 'NEW_NAME'),
    ]


def test_delete_feature(root: gws.Root):
    res = u.api_request(root, 'editGetFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', featureUid='2'))
    props = res.json['feature']

    res = u.api_request(root, 'editDeleteFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', feature=props))
    assert res.status_code == 200

    assert u.pg.rows('SELECT id,name FROM plain WHERE id=2') == []
