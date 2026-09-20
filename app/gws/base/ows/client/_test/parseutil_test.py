import gws
import gws.test.util as u
import gws.lib.crs
import gws.lib.xmlx as xmlx
import gws.base.ows.client.parseutil as pu


def _el(xml):
    return xmlx.from_string(xml, gws.XmlOptions(compactWhitespace=True, removeNamespaces=True))


def test_service_operations_wms():
    el = _el("""
        <WMS_Capabilities>
            <Capability>
                <Request>
                    <GetCapabilities>
                        <Format>text/xml</Format>
                        <DCPType><HTTP>
                            <Get><OnlineResource xlink:href="http://host/wms?" xmlns:xlink="http://www.w3.org/1999/xlink"/></Get>
                            <Post><OnlineResource xlink:href="http://host/wms_post" xmlns:xlink="http://www.w3.org/1999/xlink"/></Post>
                        </HTTP></DCPType>
                    </GetCapabilities>
                    <GetMap>
                        <Format>image/png</Format>
                        <Format>image/jpeg</Format>
                        <DCPType><HTTP>
                            <Get><OnlineResource xlink:href="http://host/wms?map=x&amp;" xmlns:xlink="http://www.w3.org/1999/xlink"/></Get>
                        </HTTP></DCPType>
                    </GetMap>
                </Request>
            </Capability>
        </WMS_Capabilities>
    """)

    ops = pu.service_operations(el)

    assert [op.verb for op in ops] == ['GetCapabilities', 'GetMap']

    assert ops[0].url == 'http://host/wms'
    assert ops[0].params == {}
    assert ops[0].postUrl == 'http://host/wms_post'
    assert ops[0].formats == ['text/xml']
    assert ops[0].allowedParameters == {}

    assert ops[1].url == 'http://host/wms'
    assert ops[1].params == {'map': 'x'}
    assert ops[1].postUrl == ''
    assert ops[1].formats == ['image/png', 'image/jpeg']


def test_service_operations_ows():
    el = _el("""
        <WFS_Capabilities xmlns:ows="http://www.opengis.net/ows/1.1" xmlns:xlink="http://www.w3.org/1999/xlink">
            <ows:OperationsMetadata>
                <ows:Operation name="GetCapabilities">
                    <ows:DCP><ows:HTTP>
                        <ows:Get xlink:href="http://host/wfs"/>
                        <ows:Post xlink:href="http://host/wfs"/>
                    </ows:HTTP></ows:DCP>
                    <ows:Parameter name="AcceptVersions">
                        <ows:AllowedValues>
                            <ows:Value>2.0.0</ows:Value>
                            <ows:Value>1.1.0</ows:Value>
                        </ows:AllowedValues>
                    </ows:Parameter>
                </ows:Operation>
                <ows:Operation name="GetFeature">
                    <ows:DCP><ows:HTTP><ows:Get xlink:href="http://host/wfs"/></ows:HTTP></ows:DCP>
                    <ows:Parameter name="outputFormat">
                        <ows:Value>text/xml; subtype=gml/3.2</ows:Value>
                        <ows:Value>application/json</ows:Value>
                    </ows:Parameter>
                </ows:Operation>
            </ows:OperationsMetadata>
        </WFS_Capabilities>
    """)

    ops = pu.service_operations(el)

    assert [op.verb for op in ops] == ['GetCapabilities', 'GetFeature']
    assert ops[0].allowedParameters == {'ACCEPTVERSIONS': ['2.0.0', '1.1.0']}
    assert ops[0].postUrl == 'http://host/wfs'
    assert ops[1].allowedParameters == {'OUTPUTFORMAT': ['text/xml; subtype=gml/3.2', 'application/json']}
    assert ops[1].formats == ['text/xml; subtype=gml/3.2', 'application/json']


def test_service_operations_empty():
    assert pu.service_operations(_el('<Capabilities/>')) == []


def test_service_metadata_wms():
    el = _el("""
        <WMS_Capabilities xmlns:xlink="http://www.w3.org/1999/xlink">
            <Service>
                <Name>WMS</Name>
                <Title>TITLE_1</Title>
                <Abstract>ABSTRACT_1</Abstract>
                <KeywordList>
                    <Keyword>KEYWORD_1</Keyword>
                    <Keyword>KEYWORD_2</Keyword>
                </KeywordList>
                <ContactInformation>
                    <ContactPersonPrimary>
                        <ContactPerson>PERSON_1</ContactPerson>
                        <ContactOrganization>ORG_1</ContactOrganization>
                    </ContactPersonPrimary>
                    <ContactPosition>POSITION_1</ContactPosition>
                    <ContactAddress>
                        <City>CITY_1</City>
                        <StateOrProvince>AREA_1</StateOrProvince>
                        <PostCode>ZIP_1</PostCode>
                        <Country>COUNTRY_1</Country>
                    </ContactAddress>
                    <ContactVoiceTelephone>PHONE_1</ContactVoiceTelephone>
                    <ContactFacsimileTelephone>FAX_1</ContactFacsimileTelephone>
                    <ContactElectronicMailAddress>EMAIL_1</ContactElectronicMailAddress>
                </ContactInformation>
                <Fees>FEES_1</Fees>
                <AccessConstraints>CONSTRAINTS_1</AccessConstraints>
            </Service>
            <ServiceMetadataURL xlink:href="http://host/meta"/>
        </WMS_Capabilities>
    """)

    md = pu.service_metadata(el)

    assert md.name == 'WMS'
    assert md.title == 'TITLE_1'
    assert md.abstract == 'ABSTRACT_1'
    assert md.keywords == ['KEYWORD_1', 'KEYWORD_2']
    assert md.fees == 'FEES_1'
    assert md.accessConstraints == 'CONSTRAINTS_1'
    assert md.contactPerson == 'PERSON_1'
    assert md.contactOrganization == 'ORG_1'
    assert md.contactPosition == 'POSITION_1'
    assert md.contactCity == 'CITY_1'
    assert md.contactArea == 'AREA_1'
    assert md.contactZip == 'ZIP_1'
    assert md.contactCountry == 'COUNTRY_1'
    assert md.contactPhone == 'PHONE_1'
    assert md.contactFax == 'FAX_1'
    assert md.contactEmail == 'EMAIL_1'
    assert md.serviceMetadataURL == 'http://host/meta'


def test_service_metadata_ows():
    el = _el("""
        <Capabilities xmlns:ows="http://www.opengis.net/ows/1.1" xmlns:xlink="http://www.w3.org/1999/xlink">
            <ows:ServiceIdentification>
                <ows:Title>TITLE_1</ows:Title>
                <ows:Abstract>ABSTRACT_1</ows:Abstract>
                <ows:Keywords>
                    <ows:Keyword>KEYWORD_1</ows:Keyword>
                    <ows:Keyword>KEYWORD_2</ows:Keyword>
                </ows:Keywords>
                <ows:Fees>FEES_1</ows:Fees>
                <ows:AccessConstraints>CONSTRAINTS_1</ows:AccessConstraints>
            </ows:ServiceIdentification>
            <ows:ServiceProvider>
                <ows:ProviderName>PROVIDER_1</ows:ProviderName>
                <ows:ProviderSite xlink:href="http://host/provider"/>
                <ows:ServiceContact>
                    <ows:IndividualName>PERSON_1</ows:IndividualName>
                    <ows:PositionName>POSITION_1</ows:PositionName>
                    <ows:ContactInfo>
                        <ows:Phone>
                            <ows:Voice>PHONE_1</ows:Voice>
                            <ows:Facsimile>FAX_1</ows:Facsimile>
                        </ows:Phone>
                        <ows:Address>
                            <ows:City>CITY_1</ows:City>
                            <ows:AdministrativeArea>AREA_1</ows:AdministrativeArea>
                            <ows:PostalCode>ZIP_1</ows:PostalCode>
                            <ows:Country>COUNTRY_1</ows:Country>
                            <ows:ElectronicMailAddress>EMAIL_1</ows:ElectronicMailAddress>
                        </ows:Address>
                    </ows:ContactInfo>
                </ows:ServiceContact>
            </ows:ServiceProvider>
        </Capabilities>
    """)

    md = pu.service_metadata(el)

    assert md.title == 'TITLE_1'
    assert md.abstract == 'ABSTRACT_1'
    assert md.keywords == ['KEYWORD_1', 'KEYWORD_2']
    assert md.fees == 'FEES_1'
    assert md.accessConstraints == 'CONSTRAINTS_1'
    assert md.contactProviderName == 'PROVIDER_1'
    assert md.contactPerson == 'PERSON_1'
    assert md.contactPosition == 'POSITION_1'
    assert md.contactCity == 'CITY_1'
    assert md.contactArea == 'AREA_1'
    assert md.contactZip == 'ZIP_1'
    assert md.contactCountry == 'COUNTRY_1'
    assert md.contactPhone == 'PHONE_1'
    assert md.contactFax == 'FAX_1'
    assert md.contactEmail == 'EMAIL_1'


def test_element_metadata():
    el = _el("""
        <Layer xmlns:xlink="http://www.w3.org/1999/xlink">
            <Name>NAME_1</Name>
            <Title>TITLE_1</Title>
            <Abstract>ABSTRACT_1</Abstract>
            <KeywordList>
                <Keyword>KEYWORD_1</Keyword>
            </KeywordList>
            <Attribution>
                <Title>ATTRIBUTION_1</Title>
            </Attribution>
            <AuthorityURL name="AUTHORITY_1">
                <OnlineResource xlink:href="http://host/authority"/>
            </AuthorityURL>
            <Identifier authority="AUTHORITY_1">IDENTIFIER_1</Identifier>
            <MetadataURL type="ISO19115:2003">
                <Format>text/xml</Format>
                <OnlineResource xlink:href="http://host/meta1"/>
            </MetadataURL>
            <MetadataURL xlink:href="http://host/meta2"/>
        </Layer>
    """)

    md = pu.element_metadata(el)

    assert md.name == 'NAME_1'
    assert md.title == 'TITLE_1'
    assert md.abstract == 'ABSTRACT_1'
    assert md.keywords == ['KEYWORD_1']
    assert md.attribution == 'ATTRIBUTION_1'
    assert md.authorityUrl == 'http://host/authority'
    assert md.authorityName == 'AUTHORITY_1'
    assert md.authorityIdentifier == 'IDENTIFIER_1'
    assert [(ml.url, ml.type, ml.format) for ml in md.metaLinks] == [
        ('http://host/meta1', 'ISO19115:2003', 'text/xml'),
        ('http://host/meta2', None, None),
    ]


def test_element_metadata_ows_identifier():
    el = _el("""
        <FeatureType xmlns:ows="http://www.opengis.net/ows/1.1">
            <ows:Identifier>NAME_1</ows:Identifier>
            <ows:Title>TITLE_1</ows:Title>
        </FeatureType>
    """)

    md = pu.element_metadata(el)

    assert md.name == 'NAME_1'
    assert md.title == 'TITLE_1'
    assert md.authorityIdentifier == 'NAME_1'


def test_wgs_extent_ex_geographic():
    el = _el("""
        <Layer>
            <EX_GeographicBoundingBox>
                <westBoundLongitude>10</westBoundLongitude>
                <eastBoundLongitude>12</eastBoundLongitude>
                <southBoundLatitude>50</southBoundLatitude>
                <northBoundLatitude>52</northBoundLatitude>
            </EX_GeographicBoundingBox>
        </Layer>
    """)
    assert pu.wgs_extent(el) == (10.0, 50.0, 12.0, 52.0)


def test_wgs_extent_wgs84_bounding_box():
    el = _el("""
        <FeatureType xmlns:ows="http://www.opengis.net/ows/1.1">
            <ows:WGS84BoundingBox>
                <ows:LowerCorner>10 50</ows:LowerCorner>
                <ows:UpperCorner>12 52</ows:UpperCorner>
            </ows:WGS84BoundingBox>
        </FeatureType>
    """)
    assert pu.wgs_extent(el) == (10.0, 50.0, 12.0, 52.0)


def test_wgs_extent_latlon_bounding_box():
    el = _el('<Layer><LatLonBoundingBox minx="10" miny="50" maxx="12" maxy="52"/></Layer>')
    assert pu.wgs_extent(el) == (10.0, 50.0, 12.0, 52.0)


def test_wgs_extent_missing():
    assert pu.wgs_extent(_el('<Layer/>')) is None


def test_wgs_extent_empty():
    el = _el('<FeatureType xmlns:ows="http://www.opengis.net/ows/1.1"><ows:WGS84BoundingBox/></FeatureType>')
    assert pu.wgs_extent(el) is None


def test_supported_crs_wms():
    el = _el("""
        <Layer>
            <CRS>EPSG:4326</CRS>
            <CRS>EPSG:3857</CRS>
            <BoundingBox CRS="EPSG:25832" minx="0" miny="0" maxx="1" maxy="1"/>
        </Layer>
    """)
    assert sorted(crs.srid for crs in pu.supported_crs(el)) == [3857, 4326, 25832]


def test_supported_crs_wms_111():
    el = _el("""
        <Layer>
            <SRS>EPSG:4326</SRS>
            <BoundingBox SRS="EPSG:25832" minx="0" miny="0" maxx="1" maxy="1"/>
        </Layer>
    """)
    assert sorted(crs.srid for crs in pu.supported_crs(el)) == [4326, 25832]


def test_supported_crs_ows():
    el = _el("""
        <FeatureType>
            <DefaultCRS>urn:ogc:def:crs:EPSG::25832</DefaultCRS>
            <OtherCRS>urn:ogc:def:crs:EPSG::4326</OtherCRS>
            <OtherCRS>NOT_A_CRS</OtherCRS>
        </FeatureType>
    """)
    assert sorted(crs.srid for crs in pu.supported_crs(el, ['EPSG:3857'])) == [3857, 4326, 25832]


def test_parse_style():
    el = _el("""
        <Style xmlns:xlink="http://www.w3.org/1999/xlink">
            <Name>STYLE_1</Name>
            <Title>TITLE_1</Title>
            <LegendURL width="20" height="20">
                <Format>image/png</Format>
                <OnlineResource xlink:type="simple" xlink:href="http://host/legend?x=1&amp;"/>
            </LegendURL>
        </Style>
    """)

    st = pu.parse_style(el)

    assert st.name == 'style_1'
    assert st.metadata.title == 'TITLE_1'
    assert st.legendUrl == 'http://host/legend?x=1'
    assert st.isDefault is False


def test_parse_style_default():
    assert pu.parse_style(_el('<Style><Name>default</Name></Style>')).isDefault is True
    assert pu.parse_style(_el('<Style><Name>ns:Default</Name></Style>')).isDefault is True
    assert pu.parse_style(_el('<Style IsDefault="true"><Name>x</Name></Style>')).isDefault is True


def test_default_style():
    st_1 = gws.SourceStyle(name='a', isDefault=False)
    st_2 = gws.SourceStyle(name='b', isDefault=True)
    assert pu.default_style([st_1, st_2]) is st_2
    assert pu.default_style([st_1]) is st_1
    assert pu.default_style([]) is None


def test_conversions():
    assert pu.to_float('1.5') == 1.5
    assert pu.to_float('') == 0.0
    assert pu.to_float(None, 2) == 2.0
    assert pu.to_int('1.5') == 1
    assert pu.to_int('') == 0
    assert pu.to_float_pair(' 1 2.5 ') == (1.0, 2.5)
