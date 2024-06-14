import gws
import gws.lib.jsonx as jsonx
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('plain', {'id': 'int primary key', 'a': 'text', 'g': 'geometry(point,3857)'})
    u.pg.insert('plain', [
        dict(id=1, a='11', g=u.ewkb('POINT(10 100)')),
        dict(id=2, a='22', g=u.ewkb('POINT(20 200)')),
        dict(id=3, a='33', g=u.ewkb('POINT(30 300)')),
    ])

    cfg = '''
        permissions.all "allow all"
        
        models+ { 
            uid "PLAIN" type "postgres" tableName "plain" 
        }
        actions+ {
            uid 'ACTION'
            type 'map'
        }
        projects+ {
            uid "A"
            permissions.all "allow all"
            map.layers+ {
                uid "LAYER"
                type "postgres"
                tableName "plain"
            }
        }            
    '''

    yield u.gws_root(cfg)


def test_get_features(root: gws.Root):
    res = u.get_request(root, '/_/mapGetFeatures/layerUid/LAYER')
    fs = res.json['features']

    assert fs[0]['attributes'] == {'id': 1, 'g': {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': [10.0, 100.0]}}}
    assert fs[1]['attributes'] == {'id': 2, 'g': {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': [20.0, 200.0]}}}
    assert fs[2]['attributes'] == {'id': 3, 'g': {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': [30.0, 300.0]}}}
