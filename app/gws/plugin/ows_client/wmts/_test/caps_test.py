import gws
import gws.test.util as u
import gws.plugin.ows_client.wmts.caps as caps


_CAPS = """<?xml version="1.0" encoding="UTF-8"?>
<Capabilities version="1.0.0"
    xmlns="http://www.opengis.net/wmts/1.0"
    xmlns:ows="http://www.opengis.net/ows/1.1"
    xmlns:xlink="http://www.w3.org/1999/xlink">
    <ows:ServiceIdentification>
        <ows:Title>SERVICE_1</ows:Title>
    </ows:ServiceIdentification>
    <ows:OperationsMetadata>
        <ows:Operation name="GetTile">
            <ows:DCP><ows:HTTP><ows:Get xlink:href="http://host/wmts?"/></ows:HTTP></ows:DCP>
        </ows:Operation>
    </ows:OperationsMetadata>
    <Contents>
        <Layer>
            <ows:Title>LAYER_1</ows:Title>
            <ows:Identifier>layer_1</ows:Identifier>
            <ows:WGS84BoundingBox>
                <ows:LowerCorner>5 47</ows:LowerCorner>
                <ows:UpperCorner>15 55</ows:UpperCorner>
            </ows:WGS84BoundingBox>
            <Style isDefault="true">
                <ows:Identifier>default</ows:Identifier>
                <LegendURL format="image/png" xlink:href="http://host/legend_1"/>
            </Style>
            <Format>image/png</Format>
            <TileMatrixSetLink>
                <TileMatrixSet>tms_1</TileMatrixSet>
            </TileMatrixSetLink>
            <ResourceURL format="image/png" resourceType="tile"
                template="http://host/tile/{TileMatrix}/{TileRow}/{TileCol}.png"/>
        </Layer>
        <TileMatrixSet>
            <ows:Identifier>tms_1</ows:Identifier>
            <ows:SupportedCRS>urn:ogc:def:crs:EPSG::3857</ows:SupportedCRS>
            <TileMatrix>
                <ows:Identifier>0</ows:Identifier>
                <ScaleDenominator>559082264.0287178</ScaleDenominator>
                <TopLeftCorner>-20037508.3427892 20037508.3427892</TopLeftCorner>
                <TileWidth>256</TileWidth>
                <TileHeight>256</TileHeight>
                <MatrixWidth>1</MatrixWidth>
                <MatrixHeight>1</MatrixHeight>
            </TileMatrix>
            <TileMatrix>
                <ows:Identifier>1</ows:Identifier>
                <ScaleDenominator>279541132.0143589</ScaleDenominator>
                <TopLeftCorner>-20037508.3427892 20037508.3427892</TopLeftCorner>
                <TileWidth>256</TileWidth>
                <TileHeight>256</TileHeight>
                <MatrixWidth>2</MatrixWidth>
                <MatrixHeight>2</MatrixHeight>
            </TileMatrix>
        </TileMatrixSet>
    </Contents>
</Capabilities>
"""


def test_parse():
    cc = caps.parse(_CAPS)

    assert cc.version == '1.0.0'
    assert cc.metadata.title == 'SERVICE_1'
    assert [op.verb for op in cc.operations] == ['GetTile']

    assert [tms.identifier for tms in cc.tileMatrixSets] == ['tms_1']
    tms = cc.tileMatrixSets[0]
    assert tms.crs.srid == 3857
    assert [tm.identifier for tm in tms.matrices] == ['0', '1']
    assert tms.matrices[0].width == 1
    assert tms.matrices[1].width == 2
    assert tms.matrices[1].tileWidth == 256
    assert tms.matrices[0].x == -20037508.3427892
    assert tms.matrices[0].y == 20037508.3427892
    assert tms.matrices[0].scale > tms.matrices[1].scale

    layer_1 = cc.sourceLayers[0]
    assert layer_1.name == 'layer_1'
    assert layer_1.title == 'LAYER_1'
    assert layer_1.isImage is True
    assert layer_1.imageFormat == 'image/png'
    assert layer_1.wgsExtent == (5.0, 47.0, 15.0, 55.0)
    assert [crs.srid for crs in layer_1.supportedCrs] == [3857]
    assert layer_1.tileMatrixIds == ['tms_1']
    assert layer_1.tileMatrixSets[0] is tms
    assert layer_1.resourceUrls == {'tile': 'http://host/tile/{TileMatrix}/{TileRow}/{TileCol}.png'}
    assert layer_1.defaultStyle.name == 'default'
    assert layer_1.legendUrl == 'http://host/legend_1'
