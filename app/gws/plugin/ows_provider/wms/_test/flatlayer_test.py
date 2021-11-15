import gws
import gws.lib.net
import gws.lib.shape
import gws.base.search.runner
import gws.base.auth.user
import gws.lib.test as test

from gws.plugin.ows_provider.wms import flatlayer

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
            access+ {{ role all type allow }}
            map.crs 'EPSG:3857'
            map.layers+ {{
                type 'wmsflat'
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
            access+ {{ role all type allow }}
            map.crs 'EPSG:3857'
            map.layers+ {{
                type 'wmsflat'
                uid 'layer_from_a_b'
                capsCacheMaxAge 0
                url {fixtures.SERVICE_URL!r}
                sourceLayers.names [A B]
            }}
        }}
    ''')

    yield root.find(uid='one.map.layer_from_a_b')


@test.fixture(scope='module')
def web_request():
    user = test.root().find(klass='gws.ext.auth.provider.system.Object').users['system']
    return gws.Data(user=user)


def search_args(layer):
    return gws.SearchArgs(
        bounds=gws.Bounds(crs='EPSG:3857', extent=[100, 200, 300, 400]),
        resolution=10,
        layers=[layer],
        shapes=[
            gws.lib.shape.from_geometry({
                'type': 'point',
                'coordinates': [100, 200]
            }, 'EPSG:3857')
        ]
    )


def render_view(layer):
    return gws.MapView(
        bounds=gws.Bounds(crs='EPSG:3857', extent=[100, 200, 300, 400]),
        dpi=0,
        size_px=(100, 100),
    )


#

def test_default_source_layer_is_root(layer_from_root: flatlayer.Object):
    assert len(layer_from_root.source_layers) == 1
    assert layer_from_root.source_layers[0].a_uid == 'root'


def test_default_bounds_are_from_root(layer_from_root: flatlayer.Object):
    assert layer_from_root.own_bounds.extent == (111.0, 222.0, 888.0, 999.0)


def test_default_render_uses_root(layer_from_root: flatlayer.Object):
    test.web_server_begin_capture()
    layer_from_root.render_box(render_view(layer_from_root))
    urls = test.web_server_end_capture()
    assert urls[0].params['layers'] == 'root'


def test_default_search_uses_queryable_layers(layer_from_root: flatlayer.Object, web_request):
    test.web_server_begin_capture()
    gws.base.search.runner.run(web_request, search_args(layer_from_root))
    urls = test.web_server_end_capture()
    assert urls[0].params['query_layers'] == 'A,C'


def test_explicit_bounds_are_combined(layer_from_a_b: flatlayer.Object):
    assert layer_from_a_b.own_bounds.extent == (100.0, 200.0, 500.0, 600.0)


def test_explicit_render_uses_configured_layers_bottom_up(layer_from_a_b: flatlayer.Object):
    test.web_server_begin_capture()
    layer_from_a_b.render_box(render_view(layer_from_a_b))
    urls = test.web_server_end_capture()
    assert urls[0].params['layers'] == 'B,A'


def test_explicit_search_user_queryable_layers(layer_from_a_b: flatlayer.Object, web_request):
    test.web_server_begin_capture()
    gws.base.search.runner.run(web_request, search_args(layer_from_a_b))
    urls = test.web_server_end_capture()
    assert urls[0].params['query_layers'] == 'A'
