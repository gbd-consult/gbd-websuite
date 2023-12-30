import gws
import gws.lib.net
import gws.base.shape
import gws.gis.crs
import gws.base.auth.user
import gws.test.util as u



@u.fixture(scope='module', autouse=True)
def configuration():
    test.mockserv.create_wms(fixtures.WMS_CONFIG)
    yield


@u.fixture(scope='module')
def layer_from_root():
    root = test.config.configure(f'''
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

    yield root.find(gws.ext.object.layer, uid='layer_from_root')


@u.fixture(scope='module')
def layer_from_a_b():
    root = test.config.configure(f'''
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


@u.fixture(scope='module')
def web_request():
    user = test.config.root().find(klass=gws.ext.object.authProvider('system')).users['system']
    return gws.Data(user=user)


def search_args(layer):
    return gws.SearchQuery(
        bounds=gws.Bounds(crs='EPSG:3857', extent=[100, 200, 300, 400]),
        resolution=10,
        layers=[layer],
        shapes=[
            gws.base.shape.from_geometry({
                'type': 'point',
                'coordinates': [100, 200]
            }, gws.gis.crs.WEBMERCATOR)
        ]
    )


def render_view(layer):
    return gws.MapView(
        bounds=gws.Bounds(crs='EPSG:3857', extent=[100, 200, 300, 400]),
        dpi=0,
        pxSize=(100, 100),
    )


#

def test_default_source_layer_is_root(layer_from_root):
    assert len(layer_from_root.sourceLayers) == 1
    assert layer_from_root.sourceLayers[0].aUid == 'root'


def test_default_bounds_are_from_root(layer_from_root):
    assert layer_from_root.own_bounds.extent == (111.0, 222.0, 888.0, 999.0)


def test_default_render_uses_root(layer_from_root):
    test.mockserv.begin_capture()
    layer_from_root.render_box(render_view(layer_from_root))
    urls = test.mockserv.end_capture()
    assert urls[0].params['layers'] == 'root'


def test_default_search_uses_queryable_layers(layer_from_root, web_request):
    test.mockserv.begin_capture()
    gws.base.search.runner.run(web_request, search_args(layer_from_root))
    urls = test.mockserv.end_capture()
    assert urls[0].params['query_layers'] == 'A,C'


def test_explicit_bounds_are_combined(layer_from_a_b):
    assert layer_from_a_b.own_bounds.extent == (100.0, 200.0, 500.0, 600.0)


def test_explicit_render_uses_configured_layers_bottom_up(layer_from_a_b):
    test.mockserv.begin_capture()
    layer_from_a_b.render_box(render_view(layer_from_a_b))
    urls = test.mockserv.end_capture()
    assert urls[0].params['layers'] == 'B,A'


def test_explicit_search_user_queryable_layers(layer_from_a_b, web_request):
    test.mockserv.begin_capture()
    gws.base.search.runner.run(web_request, search_args(layer_from_a_b))
    urls = test.mockserv.end_capture()
    assert urls[0].params['query_layers'] == 'A'
