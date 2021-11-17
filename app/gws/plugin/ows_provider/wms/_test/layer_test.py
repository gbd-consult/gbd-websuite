import gws
import gws.lib.net
import gws.gis.shape
import gws.base.search.runner
import gws.base.auth.user
import gws.lib.test as test

from gws.plugin.ows_provider.wms import layer

from gws.plugin.ows_provider.wms._test import fixtures


@test.fixture(scope='module', autouse=True)
def configuration():
    test.setup()
    test.web_server_create_wms(fixtures.WMS_CONFIG)
    yield
    test.teardown()


@test.fixture(scope='module')
def layer_from_root():
    root = test.configure_and_reload(f'''
        projects+ {{
            uid 'one'
            map.crs 'EPSG:3857'
            map.layers+ {{
                type 'wms'
                uid 'layer_from_root'
                capsCacheMaxAge 0
                url {fixtures.SERVICE_URL!r}
            }}
        }}
    ''')

    yield root.find(uid='one.map.layer_from_root')


@test.fixture(scope='module')
def layer_from_a_b():
    root = test.configure_and_reload(f'''
        projects+ {{
            uid 'one'
            map.crs 'EPSG:3857'
            map.layers+ {{
                type 'wms'
                uid 'layer_from_a_b'
                capsCacheMaxAge 0
                url {fixtures.SERVICE_URL!r}
                rootLayers.names [A B]
            }}
        }}
    ''')

    yield root.find(uid='one.map.layer_from_a_b')


@test.fixture(scope='module')
def layer_without_b():
    root = test.configure_and_reload(f'''
        projects+ {{
            uid 'one'
            map.crs 'EPSG:3857'
            map.layers+ {{
                type 'wms'
                uid 'layer_without_b'
                capsCacheMaxAge 0
                url {fixtures.SERVICE_URL!r}
                excludeLayers.names [B]
            }}
        }}
    ''')

    yield root.find(uid='one.map.layer_without_b')


#


def layer_tree(root):
    def _enum(obj):
        sub = ' '.join(_enum(s) for s in obj.layers)
        name = obj.title or obj.uid
        return f'{name}({sub})' if sub else name

    return _enum(root.find('gws.base.map'))


#

def test_default_layer_tree(layer_from_root: layer.Object):
    tree = layer_tree(layer_from_root.root)
    assert tree == 'one.map(one.map.layer_from_root(root_TITLE(A_TITLE(A1_TITLE) B_TITLE(B1_TITLE) C_TITLE(C1_TITLE C2_TITLE))))'


def test_filtered_layer_tree(layer_from_a_b: layer.Object):
    tree = layer_tree(layer_from_a_b.root)
    assert tree == 'one.map(one.map.layer_from_a_b(A_TITLE(A1_TITLE) B_TITLE(B1_TITLE)))'


def test_excluded_layer_tree(layer_without_b: layer.Object):
    tree = layer_tree(layer_without_b.root)
    assert tree == 'one.map(one.map.layer_without_b(root_TITLE(A_TITLE(A1_TITLE) C_TITLE(C1_TITLE C2_TITLE))))'
