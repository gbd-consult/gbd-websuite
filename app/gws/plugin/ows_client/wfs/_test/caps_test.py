import gws
import gws.test.util as u
import gws.plugin.ows_client.wfs.caps as caps


_CAPS_200 = """<?xml version="1.0" encoding="UTF-8"?>
<wfs:WFS_Capabilities version="2.0.0"
    xmlns:wfs="http://www.opengis.net/wfs/2.0"
    xmlns:ows="http://www.opengis.net/ows/1.1"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:ns_1="http://host/ns_1">
    <ows:ServiceIdentification>
        <ows:Title>SERVICE_1</ows:Title>
        <ows:Abstract>ABSTRACT_1</ows:Abstract>
    </ows:ServiceIdentification>
    <ows:ServiceProvider>
        <ows:ProviderName>PROVIDER_1</ows:ProviderName>
    </ows:ServiceProvider>
    <ows:OperationsMetadata>
        <ows:Operation name="GetCapabilities">
            <ows:DCP><ows:HTTP><ows:Get xlink:href="http://host/wfs"/></ows:HTTP></ows:DCP>
        </ows:Operation>
        <ows:Operation name="GetFeature">
            <ows:DCP><ows:HTTP>
                <ows:Get xlink:href="http://host/wfs"/>
                <ows:Post xlink:href="http://host/wfs"/>
            </ows:HTTP></ows:DCP>
            <ows:Parameter name="outputFormat">
                <ows:AllowedValues>
                    <ows:Value>application/gml+xml; version=3.2</ows:Value>
                </ows:AllowedValues>
            </ows:Parameter>
        </ows:Operation>
    </ows:OperationsMetadata>
    <wfs:FeatureTypeList>
        <wfs:FeatureType>
            <wfs:Name>ns_1:type_1</wfs:Name>
            <wfs:Title>TYPE_1</wfs:Title>
            <wfs:Abstract>ABSTRACT_2</wfs:Abstract>
            <ows:Keywords><ows:Keyword>KEYWORD_1</ows:Keyword></ows:Keywords>
            <wfs:DefaultCRS>urn:ogc:def:crs:EPSG::25832</wfs:DefaultCRS>
            <wfs:OtherCRS>urn:ogc:def:crs:EPSG::4326</wfs:OtherCRS>
            <ows:WGS84BoundingBox>
                <ows:LowerCorner>5 47</ows:LowerCorner>
                <ows:UpperCorner>15 55</ows:UpperCorner>
            </ows:WGS84BoundingBox>
        </wfs:FeatureType>
        <wfs:FeatureType>
            <wfs:Name>ns_1:type_2</wfs:Name>
            <wfs:DefaultCRS>urn:ogc:def:crs:EPSG::3857</wfs:DefaultCRS>
        </wfs:FeatureType>
    </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>
"""

_CAPS_110 = """<?xml version="1.0" encoding="UTF-8"?>
<WFS_Capabilities version="1.1.0"
    xmlns="http://www.opengis.net/wfs"
    xmlns:ows="http://www.opengis.net/ows"
    xmlns:xlink="http://www.w3.org/1999/xlink">
    <ows:ServiceIdentification>
        <ows:Title>SERVICE_1</ows:Title>
    </ows:ServiceIdentification>
    <ows:OperationsMetadata>
        <ows:Operation name="GetFeature">
            <ows:DCP><ows:HTTP><ows:Get xlink:href="http://host/wfs?"/></ows:HTTP></ows:DCP>
            <ows:Parameter name="resultType">
                <ows:Value>results</ows:Value>
                <ows:Value>hits</ows:Value>
            </ows:Parameter>
        </ows:Operation>
    </ows:OperationsMetadata>
    <FeatureTypeList>
        <FeatureType>
            <Name>type_1</Name>
            <Title>TYPE_1</Title>
            <DefaultSRS>urn:ogc:def:crs:EPSG::31467</DefaultSRS>
            <OtherSRS>EPSG:4326</OtherSRS>
            <ows:WGS84BoundingBox>
                <ows:LowerCorner>5 47</ows:LowerCorner>
                <ows:UpperCorner>15 55</ows:UpperCorner>
            </ows:WGS84BoundingBox>
        </FeatureType>
    </FeatureTypeList>
</WFS_Capabilities>
"""


def test_parse_200():
    cc = caps.parse(_CAPS_200)

    assert cc.version == '2.0.0'
    assert cc.metadata.title == 'SERVICE_1'
    assert cc.metadata.abstract == 'ABSTRACT_1'
    assert cc.metadata.contactProviderName == 'PROVIDER_1'

    assert [op.verb for op in cc.operations] == ['GetCapabilities', 'GetFeature']
    assert cc.operations[1].postUrl == 'http://host/wfs'
    assert cc.operations[1].formats == ['application/gml+xml; version=3.2']

    assert [sl.name for sl in cc.sourceLayers] == ['ns_1:type_1', 'ns_1:type_2']

    type_1 = cc.sourceLayers[0]
    assert type_1.title == 'TYPE_1'
    assert type_1.metadata.abstract == 'ABSTRACT_2'
    assert type_1.metadata.keywords == ['KEYWORD_1']
    assert type_1.isQueryable is True
    assert sorted(crs.srid for crs in type_1.supportedCrs) == [4326, 25832]
    assert type_1.wgsExtent == (5.0, 47.0, 15.0, 55.0)

    type_2 = cc.sourceLayers[1]
    assert type_2.title == 'type_2'
    assert [crs.srid for crs in type_2.supportedCrs] == [3857]
    assert type_2.wgsExtent == gws.lib.crs.WGS84.extent


def test_parse_110():
    cc = caps.parse(_CAPS_110)

    assert cc.version == '1.1.0'
    assert cc.metadata.title == 'SERVICE_1'
    assert [op.verb for op in cc.operations] == ['GetFeature']
    assert cc.operations[0].url == 'http://host/wfs'
    assert cc.operations[0].allowedParameters == {'RESULTTYPE': ['results', 'hits']}

    type_1 = cc.sourceLayers[0]
    assert type_1.name == 'type_1'
    assert type_1.title == 'TYPE_1'
    assert sorted(crs.srid for crs in type_1.supportedCrs) == [4326, 31467]
    assert type_1.wgsExtent == (5.0, 47.0, 15.0, 55.0)
