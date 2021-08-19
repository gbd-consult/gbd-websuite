import gws
import gws.lib.net
import gws.lib.shape
import gws.lib.test as test
from gws.plugin.ows_provider.wms import flatlayer

SERVICE_URL = test.web_server_url('WMS')

REQUEST_CAPS = f"""
        <Request>
            <GetCapabilities>
                <Format>text/xml</Format>
                <DCPType>
                    <HTTP>
                        <Get>
                            <OnlineResource xlink:href="{SERVICE_URL}/GetCapabilities"/>
                        </Get>
                    </HTTP>
                </DCPType>
            </GetCapabilities>
            <GetMap>
                <DCPType>
                    <HTTP>
                        <Get>
                            <OnlineResource xlink:href="{SERVICE_URL}/GetMap"/>
                        </Get>
                    </HTTP>
                </DCPType>
            </GetMap>
            <GetFeatureInfo>
                <DCPType>
                    <HTTP>
                        <Get>
                            <OnlineResource xlink:href="{SERVICE_URL}/GetFeatureInfo"/>
                        </Get>
                    </HTTP>
                </DCPType>
            </GetFeatureInfo>
            <sld:GetLegendGraphic>
                <Format>image/jpeg</Format>
                <Format>image/png</Format>
                <DCPType>
                    <HTTP>
                        <Get>
                            <OnlineResource xlink:href="{SERVICE_URL}/ServiceLegend"/>
                        </Get>
                    </HTTP>
                </DCPType>
            </sld:GetLegendGraphic>
        </Request>
"""

CAPS_1 = f"""
    <WMS_Capabilities version="1.3.0">
        <Service>
            <Name>WMS</Name>
        </Service>
        <Capability>
            {REQUEST_CAPS}
    
            <Layer queryable="1">
                <Name>root</Name>
                <CRS>EPSG:3857</CRS>
                <BoundingBox CRS="EPSG:3857" minx="1" miny="2" maxx="3" maxy="4"/>
                
                
                <Layer queryable="1">
                    <Name>AAA</Name>
                    <BoundingBox CRS="EPSG:3857" minx="1000" miny="1200" maxx="1300" maxy="1400"/>
                    <Style>
                        <Name>default</Name>
                        <LegendURL>
                            <Format>image/png</Format>
                            <OnlineResource xlink:href="{SERVICE_URL}/LegendA"/>
                        </LegendURL>
                    </Style>
                </Layer>
    
                <Layer queryable="0">
                    <Name>BBB</Name>
                    <BoundingBox CRS="EPSG:3857" minx="2000" miny="2100" maxx="2200" maxy="2300"/>
                    <Style>
                        <Name>default</Name>
                        <LegendURL>
                            <Format>image/png</Format>
                            <OnlineResource xlink:href="{SERVICE_URL}/LegendB"/>
                        </LegendURL>
                    </Style>
                    
                    <Layer queryable="1">
                        <Name>BBB_2</Name>
                    </Layer>

                </Layer>
    
                <Layer queryable="1">
                    <Name>CCC</Name>
                    <BoundingBox CRS="EPSG:3857" minx="3000" miny="3100" maxx="3200" maxy="3300"/>
                    <Style>
                        <Name>default</Name>
                        <LegendURL>
                            <Format>image/png</Format>
                            <OnlineResource xlink:href="{SERVICE_URL}/LegendC"/>
                        </LegendURL>
                    </Style>
                </Layer>
            </Layer>
        </Capability>
    </WMS_Capabilities>    
"""


@test.fixture(scope='module', autouse=True)
def configuration():
    test.setup()
    test.web_server_poke('WMS', CAPS_1)
    yield
    test.teardown()


@test.fixture(scope='module')
def layer_1():
    root = test.configure(f'''
        projects+ {{
            uid 'one'
            map.crs 'EPSG:3857'
            map.layers+ {{
                type 'wmsflat'
                url {SERVICE_URL!r}
            }}
        }}
    ''')

    yield root.find('gws.ext.layer.wmsflat')


@test.fixture(scope='module')
def layer_2():
    root = test.configure(f'''
        projects+ {{
            uid 'one'
            map.crs 'EPSG:3857'
            map.layers+ {{
                type 'wmsflat'
                url {SERVICE_URL!r}
                sourceLayers.level 2
            }}
        }}
    ''')

    yield root.find('gws.ext.layer.wmsflat')


@test.fixture
def search_args():
    yield gws.SearchArgs(
        bounds=gws.Bounds(crs='EPSG:3857', extent=[100, 200, 300, 400]),
        resolution=10,
        shapes=[
            gws.lib.shape.from_geometry({
                'type': 'point',
                'coordinates': [100, 200]
            }, 'EPSG:3857')
        ]
    )


#

def test_default_source_layer_is_root(layer_1: flatlayer.Object):
    assert len(layer_1.source_layers) == 1
    assert layer_1.source_layers[0].a_uid == 'root'


def test_default_bounds(layer_1: flatlayer.Object):
    assert layer_1.own_bounds.extent == (1.0, 2.0, 3.0, 4.0)


def test_root_search(layer_1: flatlayer.Object, search_args):
    test.web_server_begin_capture()
    layer_1.search_providers[0].run(search_args)
    urls = test.web_server_end_capture()
    assert urls[0].params['query_layers'] == 'root'


def test_combined_bounds(layer_2: flatlayer.Object):
    assert layer_2.own_bounds.extent == (1000.0, 1200.0, 3200.0, 3300.0)


def test_combined_search(layer_2: flatlayer.Object, search_args):
    test.web_server_begin_capture()
    layer_2.search_providers[0].run(search_args)
    urls = test.web_server_end_capture()
    assert urls[0].params['query_layers'] == 'AAA,CCC'
