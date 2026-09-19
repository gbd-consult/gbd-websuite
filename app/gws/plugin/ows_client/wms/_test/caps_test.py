import gws
import gws.test.util as u
import gws.plugin.ows_client.wms.caps as caps


_CAPS_130 = """<?xml version="1.0" encoding="UTF-8"?>
<WMS_Capabilities version="1.3.0" xmlns="http://www.opengis.net/wms" xmlns:xlink="http://www.w3.org/1999/xlink">
    <Service>
        <Name>WMS</Name>
        <Title>SERVICE_1</Title>
    </Service>
    <Capability>
        <Request>
            <GetCapabilities>
                <Format>text/xml</Format>
                <DCPType><HTTP><Get><OnlineResource xlink:href="http://host/wms?"/></Get></HTTP></DCPType>
            </GetCapabilities>
            <GetMap>
                <Format>image/png</Format>
                <DCPType><HTTP><Get><OnlineResource xlink:href="http://host/wms?"/></Get></HTTP></DCPType>
            </GetMap>
        </Request>
        <Layer>
            <Title>ROOT_1</Title>
            <CRS>EPSG:4326</CRS>
            <CRS>EPSG:3857</CRS>
            <EX_GeographicBoundingBox>
                <westBoundLongitude>0</westBoundLongitude>
                <eastBoundLongitude>10</eastBoundLongitude>
                <southBoundLatitude>0</southBoundLatitude>
                <northBoundLatitude>10</northBoundLatitude>
            </EX_GeographicBoundingBox>
            <Attribution><Title>ATTRIBUTION_1</Title></Attribution>
            <Style>
                <Name>root_style</Name>
                <Title>ROOT_STYLE</Title>
            </Style>
            <Layer queryable="1">
                <Name>layer_1</Name>
                <Title>LAYER_1</Title>
                <CRS>EPSG:25832</CRS>
                <EX_GeographicBoundingBox>
                    <westBoundLongitude>1</westBoundLongitude>
                    <eastBoundLongitude>2</eastBoundLongitude>
                    <southBoundLatitude>3</southBoundLatitude>
                    <northBoundLatitude>4</northBoundLatitude>
                </EX_GeographicBoundingBox>
                <Style>
                    <Name>default</Name>
                    <Title>DEFAULT</Title>
                    <LegendURL><Format>image/png</Format><OnlineResource xlink:href="http://host/legend_1"/></LegendURL>
                </Style>
                <MinScaleDenominator>1000</MinScaleDenominator>
                <MaxScaleDenominator>50000.5</MaxScaleDenominator>
            </Layer>
            <Layer>
                <Name>group_1</Name>
                <Title>GROUP_1</Title>
                <Layer queryable="0">
                    <Name>layer_2</Name>
                    <Title>LAYER_2</Title>
                </Layer>
                <Layer>
                    <Title>UNNAMED</Title>
                </Layer>
            </Layer>
        </Layer>
    </Capability>
</WMS_Capabilities>
"""

_CAPS_111 = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE WMT_MS_Capabilities SYSTEM "http://schemas.opengis.net/wms/1.1.1/capabilities_1_1_1.dtd">
<WMT_MS_Capabilities version="1.1.1">
    <Service>
        <Name>OGC:WMS</Name>
        <Title>SERVICE_1</Title>
    </Service>
    <Capability>
        <Request>
            <GetMap>
                <Format>image/png</Format>
                <DCPType><HTTP><Get><OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://host/wms?"/></Get></HTTP></DCPType>
            </GetMap>
        </Request>
        <Layer>
            <Title>ROOT_1</Title>
            <SRS>EPSG:4326</SRS>
            <LatLonBoundingBox minx="0" miny="0" maxx="10" maxy="10"/>
            <Layer queryable="1">
                <Name>layer_1</Name>
                <Title>LAYER_1</Title>
                <SRS>EPSG:31467</SRS>
                <BoundingBox SRS="EPSG:31467" minx="1" miny="2" maxx="3" maxy="4"/>
            </Layer>
        </Layer>
    </Capability>
</WMT_MS_Capabilities>
"""


def test_parse_130():
    cc = caps.parse(_CAPS_130)

    assert cc.version == '1.3.0'
    assert cc.metadata.title == 'SERVICE_1'
    assert [op.verb for op in cc.operations] == ['GetCapabilities', 'GetMap']

    root = cc.sourceLayers[0]
    assert root.title == 'ROOT_1'
    assert root.name == ''
    assert root.isGroup is True
    assert root.isImage is False
    assert root.isQueryable is False
    assert root.wgsExtent == (0.0, 0.0, 10.0, 10.0)
    assert sorted(crs.srid for crs in root.supportedCrs) == [3857, 4326]
    assert root.metadata.attribution == 'ATTRIBUTION_1'
    assert [c.name for c in root.layers] == ['layer_1', 'group_1']

    layer_1 = root.layers[0]
    assert layer_1.title == 'LAYER_1'
    assert layer_1.isQueryable is True
    assert layer_1.isImage is True
    assert layer_1.isGroup is False
    assert layer_1.wgsExtent == (1.0, 3.0, 2.0, 4.0)
    assert sorted(crs.srid for crs in layer_1.supportedCrs) == [3857, 4326, 25832]
    assert layer_1.scaleRange == [1000, 50000]
    assert layer_1.metadata.attribution == 'ATTRIBUTION_1'
    assert sorted(st.name for st in layer_1.styles) == ['default', 'root_style']
    assert layer_1.defaultStyle.name == 'default'
    assert layer_1.legendUrl == 'http://host/legend_1'

    group_1 = root.layers[1]
    assert group_1.isGroup is True
    assert group_1.wgsExtent == (0.0, 0.0, 10.0, 10.0)
    assert [c.title for c in group_1.layers] == ['LAYER_2', 'UNNAMED']

    layer_2 = group_1.layers[0]
    assert layer_2.isQueryable is False
    assert layer_2.isImage is True
    assert layer_2.scaleRange is None

    unnamed = group_1.layers[1]
    assert unnamed.name == ''
    assert unnamed.isImage is False
    assert unnamed.isQueryable is False


def test_parse_130_bottom_first():
    cc = caps.parse(_CAPS_130, bottom_first=True)
    root = cc.sourceLayers[0]
    assert [c.name for c in root.layers] == ['group_1', 'layer_1']


def test_parse_111():
    cc = caps.parse(_CAPS_111)

    assert cc.version == '1.1.1'
    assert cc.metadata.name == 'OGC:WMS'
    assert [op.verb for op in cc.operations] == ['GetMap']
    assert cc.operations[0].url == 'http://host/wms'

    root = cc.sourceLayers[0]
    assert root.wgsExtent == (0.0, 0.0, 10.0, 10.0)
    assert [crs.srid for crs in root.supportedCrs] == [4326]

    layer_1 = root.layers[0]
    assert layer_1.name == 'layer_1'
    assert layer_1.isQueryable is True
    assert layer_1.wgsExtent == (0.0, 0.0, 10.0, 10.0)
    assert sorted(crs.srid for crs in layer_1.supportedCrs) == [4326, 31467]
