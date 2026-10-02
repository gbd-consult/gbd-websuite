import gws
import gws.base.edit.helper
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('plain', {'id': 'int primary key', 'name': 'text', 'g': 'geometry(point,3857)'})
    u.pg.insert('plain', [
        dict(id=1, name='a11', g=u.pg.ewkb('POINT(10 100)')),
        dict(id=2, name='a22', g=u.pg.ewkb('POINT(20 200)')),
        dict(id=3, name='a33', g=u.pg.ewkb('POINT(30 300)')),
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
        projects+ {
            uid "B"
            map.crs 3857
            map.layers+ { uid "LAYER_B" type "postgres" tableName "plain" }
            models+ {
                uid "MODEL_PLAIN_B"
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
    res = u.http.api(root, 'editGetModels', dict(projectUid='A'))
    ms = res.json['models']
    assert ms[0]['uid'] == 'MODEL_PLAIN'
    assert len(ms[0]['fields']) == 3


def test_get_feature(root: gws.Root):
    res = u.http.api(root, 'editGetFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', featureUid='2'))
    props = res.json['feature']
    assert props['attributes']['id'] == 2
    assert props['attributes']['name'] == 'a22'


def test_get_features_extent_uses_the_project_map_crs(root: gws.Root):
    res = u.http.api(root, 'editGetFeatures', dict(projectUid='B', modelUids=['MODEL_PLAIN_B'], extent=[0, 0, 15, 150]))
    assert res.status_code == 200
    assert [f['attributes']['id'] for f in res.json['features']] == [1]


def test_get_features_extent_without_crs_and_project_map_returns_nothing(root: gws.Root):
    assert u.cast(gws.Project, root.get('A')).map is None

    res = u.http.api(root, 'editGetFeatures', dict(projectUid='A', modelUids=['MODEL_PLAIN'], extent=[0, 0, 15, 150]))
    assert res.status_code == 200
    assert res.json['features'] == []


def test_feature_list_to_props_without_project(root: gws.Root):
    h = u.cast(gws.base.edit.helper.Object, root.app.helper('edit'))
    model = u.cast(gws.Model, root.get('MODEL_PLAIN'))

    fs = model.get_features([1], u.model.context())
    ps = h.feature_list_to_props(fs, u.model.context(project=None))
    assert ps[0].attributes['id'] == 1
    assert ps[0].attributes['name'] == 'a11'
    assert ps[0].views['title'] == '--1/a11--'


def test_write_feature(root: gws.Root):
    res = u.http.api(root, 'editGetFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', featureUid='2'))
    props = res.json['feature']
    props['attributes']['name'] = 'a22-new'

    res = u.http.api(root, 'editWriteFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', feature=props))
    assert res.status_code == 200

    assert u.pg.rows('SELECT id,name FROM plain ORDER BY id') == [
        (1, 'a11'),
        (2, 'a22-new'),
        (3, 'a33'),
    ]


def test_create_feature(root: gws.Root):
    res = u.http.api(root, 'editGetFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', featureUid='2'))
    props = res.json['feature']
    props['attributes'] = {'id': 777, 'name': 'NEW_NAME'}
    props['isNew'] = True

    res = u.http.api(root, 'editWriteFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', feature=props))
    assert res.status_code == 200

    assert u.pg.rows('SELECT id,name FROM plain WHERE id=777') == [
        (777, 'NEW_NAME'),
    ]


def test_delete_feature(root: gws.Root):
    res = u.http.api(root, 'editGetFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', featureUid='2'))
    props = res.json['feature']

    res = u.http.api(root, 'editDeleteFeature', dict(projectUid='A', modelUid='MODEL_PLAIN', feature=props))
    assert res.status_code == 200

    assert u.pg.rows('SELECT id,name FROM plain WHERE id=2') == []
