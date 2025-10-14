import gws
import gws.test.util as u
import gws.lib.crs
import gws.base.ows.client.featureinfo as featureinfo


def test_parse_msgmloutput():
    """Test MapServer msGMLOutput format parsing."""
    xml = """
    <msGMLOutput xmlns:gml="http://www.opengis.net/gml">
        <roads_layer>
            <gml:name>Road Network</gml:name>
            <road_1 fid="roads.1">
                <gml:boundedBy>
                    <gml:Box srsName="EPSG:4326">
                        <gml:coordinates>-71.123,42.234 -71.122,42.235</gml:coordinates>
                    </gml:Box>
                </gml:boundedBy>
                <GEOMETRY>
                    <gml:LineString srsName="EPSG:4326">
                        <gml:coordinates>-71.123,42.234 -71.122,42.235</gml:coordinates>
                    </gml:LineString>
                </GEOMETRY>
                <name>Main Street</name>
                <type>residential</type>
                <speed_limit>25</speed_limit>
            </road_1>
            <road_2 fid="roads.2">
                <name>Highway 101</name>
                <type>highway</type>
                <speed_limit>65</speed_limit>
            </road_2>
        </roads_layer>
        <buildings_layer>
            <gml:name>Buildings</gml:name>
            <building_1 id="bldg.100">
                <name>City Hall</name>
                <height>45</height>
                <use>government</use>
            </building_1>
        </buildings_layer>
    </msGMLOutput>
    """

    rs = featureinfo.parse(xml)

    assert len(rs) == 3

    # Road features
    road1 = next(r for r in rs if r.uid == 'roads.1')
    assert road1.meta['layerName'] == 'Road Network'
    assert road1.attributes['name'] == 'Main Street'
    assert road1.attributes['type'] == 'residential'
    assert road1.shape is not None

    road2 = next(r for r in rs if r.uid == 'roads.2')
    assert road2.meta['layerName'] == 'Road Network'
    assert road2.attributes['name'] == 'Highway 101'
    assert road2.attributes['speed_limit'] == '65'

    # Building feature
    bldg = next(r for r in rs if r.uid == 'bldg.100')
    assert bldg.meta['layerName'] == 'Buildings'
    assert bldg.attributes['name'] == 'City Hall'


def test_parse_featurecollection():
    """Test OGC FeatureCollection format parsing."""
    xml = """
    <wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs"
                            xmlns:gml="http://www.opengis.net/gml"
                            xmlns:app="http://example.com/app">
        <wfs:member>
            <app:buildings gml:id="buildings.123">
                <app:name>City Hall</app:name>
                <app:height>45.5</app:height>
                <app:use>government</app:use>
                <app:geometry>
                    <gml:Polygon srsName="EPSG:4326">
                        <gml:outerBoundaryIs>
                            <gml:LinearRing>
                                <gml:coordinates>-71.1,42.2 -71.09,42.2 -71.09,42.21 -71.1,42.21 -71.1,42.2</gml:coordinates>
                            </gml:LinearRing>
                        </gml:outerBoundaryIs>
                    </gml:Polygon>
                </app:geometry>
            </app:buildings>
        </wfs:member>
        <wfs:member>
            <app:roads gml:id="roads.456">
                <app:name>Main Street</app:name>
                <app:type>residential</app:type>
                <app:lanes>2</app:lanes>
            </app:roads>
        </wfs:member>
        <wfs:member>
            <app:parks gml:id="parks.789">
                <app:name>Central Park</app:name>
                <app:area>125.5</app:area>
                <app:facilities>
                    <app:playground>yes</app:playground>
                    <app:parking>no</app:parking>
                </app:facilities>
            </app:parks>
        </wfs:member>
    </wfs:FeatureCollection>
    """

    rs = featureinfo.parse(xml)

    assert len(rs) == 3

    building = next(r for r in rs if r.uid == 'buildings.123')
    assert building.meta['layerName'] == 'buildings'
    assert building.attributes['name'] == 'City Hall'
    assert building.attributes['height'] == '45.5'
    assert building.shape is not None

    road = next(r for r in rs if r.uid == 'roads.456')
    assert road.meta['layerName'] == 'roads'
    assert road.attributes['name'] == 'Main Street'
    assert road.attributes['lanes'] == '2'

    park = next(r for r in rs if r.uid == 'parks.789')
    assert park.meta['layerName'] == 'parks'
    assert park.attributes['name'] == 'Central Park'
    assert park.attributes['facilities.playground'] == 'yes'
    assert park.attributes['facilities.parking'] == 'no'


def test_parse_getfeatureinforesponse():
    """Test GeoServer GetFeatureInfoResponse format parsing."""
    xml = """
    <GetFeatureInfoResponse>
        <Layer name="ne:populated_places">
            <Feature id="populated_places.1">
                <Attribute name="name" value="Boston"/>
                <Attribute name="pop_max" value="4628910"/>
                <Attribute name="adm0name" value="United States of America"/>
                <Attribute name="geometry" value="POINT(-71.0275 42.3584)"/>
            </Feature>
            <Feature id="populated_places.2">
                <Attribute name="name" value="Cambridge"/>
                <Attribute name="pop_max" value="105162"/>
                <Attribute name="adm0name" value="United States of America"/>
                <Attribute name="geometry" value="POINT(-71.1056 42.3736)"/>
            </Feature>
        </Layer>
        <Layer name="ne:roads">
            <Feature id="roads.500">
                <Attribute name="name" value="Interstate 95"/>
                <Attribute name="type" value="highway"/>
                <Attribute name="lanes" value="6"/>
                <Attribute name="geometry" value="LINESTRING(-71.0 42.3, -71.1 42.4)"/>
            </Feature>
            <Feature id="roads.501">
                <Attribute name="name" value="Commonwealth Ave"/>
                <Attribute name="type" value="arterial"/>
                <Attribute name="lanes" value="4"/>
            </Feature>
        </Layer>
    </GetFeatureInfoResponse>
    """

    rs = featureinfo.parse(xml, default_crs=gws.lib.crs.WGS84)

    assert len(rs) == 4

    boston = next(r for r in rs if r.uid == 'populated_places.1')
    assert boston.meta['layerName'] == 'ne:populated_places'
    assert boston.attributes['name'] == 'Boston'
    assert boston.attributes['pop_max'] == '4628910'
    assert boston.shape is not None

    cambridge = next(r for r in rs if r.uid == 'populated_places.2')
    assert cambridge.attributes['name'] == 'Cambridge'
    assert cambridge.attributes['pop_max'] == '105162'

    highway = next(r for r in rs if r.uid == 'roads.500')
    assert highway.meta['layerName'] == 'ne:roads'
    assert highway.attributes['name'] == 'Interstate 95'
    assert highway.attributes['lanes'] == '6'
    assert highway.shape is not None

    avenue = next(r for r in rs if r.uid == 'roads.501')
    assert avenue.attributes['name'] == 'Commonwealth Ave'
    assert avenue.attributes['type'] == 'arterial'


def test_parse_featureinforesponse():
    """Test ArcGIS FeatureInfoResponse format parsing."""
    xml = """
    <FeatureInfoResponse xmlns:esri_wms="http://www.esri.com/wms">
        <FIELDS objectid="1001" shape="polygon" area="15632.45" landuse="residential" zone="R1"/>
        <FIELDS objectid="1002" shape="point" population="25000" city="Springfield" state="MA"/>
        <FIELDS fid="1003" shape="linestring" name="Route 66" highway_type="interstate" lanes="4"/>
        <FIELDS id="1004" area="8500.25" landuse="commercial" zone="C2"/>
    </FeatureInfoResponse>
    """

    rs = featureinfo.parse(xml)

    assert len(rs) == 4

    residential = next(r for r in rs if r.uid == '1001')
    assert residential.attributes['area'] == '15632.45'
    assert residential.attributes['landuse'] == 'residential'
    assert residential.attributes['zone'] == 'R1'
    assert 'shape' not in residential.attributes

    city = next(r for r in rs if r.uid == '1002')
    assert city.attributes['population'] == '25000'
    assert city.attributes['city'] == 'Springfield'
    assert city.attributes['state'] == 'MA'

    highway = next(r for r in rs if r.uid == '1003')
    assert highway.attributes['name'] == 'Route 66'
    assert highway.attributes['lanes'] == '4'

    commercial = next(r for r in rs if r.uid == '1004')
    assert commercial.attributes['landuse'] == 'commercial'
    assert commercial.attributes['zone'] == 'C2'


def test_parse_geobak():
    """Test GeoBAK format parsing."""
    xml = """
    <geobak_20:Sachdatenabfrage xmlns:geobak_20="http://www.geobak.sachsen.de/20">
        <geobak_20:Kartenebene>Flurstücke</geobak_20:Kartenebene>
        <geobak_20:Inhalt>
            <geobak_20:Datensatz>
                <geobak_20:Attribut>
                    <geobak_20:Name>Flurstücksnummer</geobak_20:Name>
                    <geobak_20:Wert>123/45</geobak_20:Wert>
                </geobak_20:Attribut>
                <geobak_20:Attribut>
                    <geobak_20:Name>Gemarkung</geobak_20:Name>
                    <geobak_20:Wert>Dresden</geobak_20:Wert>
                </geobak_20:Attribut>
                <geobak_20:Attribut>
                    <geobak_20:Name>Fläche</geobak_20:Name>
                    <geobak_20:Wert>1250.5</geobak_20:Wert>
                </geobak_20:Attribut>
            </geobak_20:Datensatz>
        </geobak_20:Inhalt>
        <geobak_20:Inhalt>
            <geobak_20:Datensatz>
                <geobak_20:Attribut>
                    <geobak_20:Name>Flurstücksnummer</geobak_20:Name>
                    <geobak_20:Wert>678/90</geobak_20:Wert>
                </geobak_20:Attribut>
                <geobak_20:Attribut>
                    <geobak_20:Name>Gemarkung</geobak_20:Name>
                    <geobak_20:Wert>Leipzig</geobak_20:Wert>
                </geobak_20:Attribut>
                <geobak_20:Attribut>
                    <geobak_20:Name>Fläche</geobak_20:Name>
                    <geobak_20:Wert>875.0</geobak_20:Wert>
                </geobak_20:Attribut>
            </geobak_20:Datensatz>
        </geobak_20:Inhalt>
    </geobak_20:Sachdatenabfrage>
    """

    rs = featureinfo.parse(xml)

    assert len(rs) == 2

    dresden_rec = next(r for r in rs if r.attributes.get('gemarkung') == 'Dresden')
    assert dresden_rec.attributes['flurstücksnummer'] == '123/45'
    assert dresden_rec.attributes['fläche'] == '1250.5'

    leipzig_rec = next(r for r in rs if r.attributes.get('gemarkung') == 'Leipzig')
    assert leipzig_rec.attributes['flurstücksnummer'] == '678/90'
    assert leipzig_rec.attributes['fläche'] == '875.0'


def test_parse_error():
    """Test parsing empty or invalid responses."""
    # Empty string
    rs = featureinfo.parse('')
    assert rs == []

    # Non-XML content
    with u.raises(featureinfo.Error):
        rs = featureinfo.parse('No features found')

    # Invalid XML
    with u.raises(featureinfo.Error):
        rs = featureinfo.parse('<invalid><xml')

    xml = """
    <UnknownFormat>
        <SomeElement>
            <attribute>value</attribute>
        </SomeElement>
    </UnknownFormat>
    """

    with u.raises(featureinfo.Error):
        rs = featureinfo.parse(xml)


def test_parse_nested_attributes():
    """Test parsing nested attributes in GML features."""
    xml = """
    <wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs"
                            xmlns:gml="http://www.opengis.net/gml"
                            xmlns:app="http://example.com/app">
        <wfs:member>
            <app:complex_feature gml:id="complex.1">
                <app:simple_attr>Simple Value</app:simple_attr>
                <app:nested_element>
                    <app:sub_attr1>Sub Value 1</app:sub_attr1>
                    <app:sub_attr2>Sub Value 2</app:sub_attr2>
                </app:nested_element>
            </app:complex_feature>
        </wfs:member>
        <wfs:member>
            <app:building gml:id="building.1">
                <app:name>Office Building</app:name>
                <app:address>
                    <app:street>123 Main St</app:street>
                    <app:city>Boston</app:city>
                    <app:contact>
                        <app:phone>555-1234</app:phone>
                        <app:email>info@example.com</app:email>
                    </app:contact>
                </app:address>
            </app:building>
        </wfs:member>
    </wfs:FeatureCollection>
    """

    rs = featureinfo.parse(xml)

    assert len(rs) == 2

    complex_rec = next(r for r in rs if r.uid == 'complex.1')
    assert complex_rec.attributes['simple_attr'] == 'Simple Value'
    assert complex_rec.attributes['nested_element.sub_attr1'] == 'Sub Value 1'
    assert complex_rec.attributes['nested_element.sub_attr2'] == 'Sub Value 2'

    building_rec = next(r for r in rs if r.uid == 'building.1')
    assert building_rec.attributes['name'] == 'Office Building'
    assert building_rec.attributes['address.street'] == '123 Main St'
    assert building_rec.attributes['address.city'] == 'Boston'
    assert building_rec.attributes['address.contact.phone'] == '555-1234'


def test_parse_with_boundedby_only():
    """Test parsing feature with only boundedBy geometry."""
    xml = """
    <msGMLOutput xmlns:gml="http://www.opengis.net/gml">
        <test_layer>
            <test_feature>
                <gml:boundedBy>
                    <gml:Box srsName="EPSG:4326">
                        <gml:coordinates>-71.123,42.234 -71.122,42.235</gml:coordinates>
                    </gml:Box>
                </gml:boundedBy>
                <name>Test Feature</name>
            </test_feature>
            <test_feature2>
                <gml:boundedBy>
                    <gml:Box srsName="EPSG:4326">
                        <gml:coordinates>-70.123,41.234 -70.122,41.235</gml:coordinates>
                    </gml:Box>
                </gml:boundedBy>
                <name>Another Feature</name>
                <type>test</type>
            </test_feature2>
        </test_layer>
    </msGMLOutput>
    """

    rs = featureinfo.parse(xml)

    assert len(rs) == 2

    feat1 = next(r for r in rs if r.attributes.get('name') == 'Test Feature')
    assert feat1.shape is not None

    feat2 = next(r for r in rs if r.attributes.get('name') == 'Another Feature')
    assert feat2.attributes['type'] == 'test'
    assert feat2.shape is not None


def test_parse_case_insensitive():
    """Test that XML parsing is case insensitive."""
    xml = """
    <FEATURECOLLECTION>
        <MEMBER>
            <BUILDING ID="1">
                <NAME>Test Building</NAME>
                <HEIGHT>50</HEIGHT>
            </BUILDING>
        </MEMBER>
        <MEMBER>
            <ROAD ID="2">
                <NAME>Main Street</NAME>
                <TYPE>residential</TYPE>
            </ROAD>
        </MEMBER>
    </FEATURECOLLECTION>
    """

    rs = featureinfo.parse(xml)

    assert len(rs) == 2

    building = next(r for r in rs if r.uid == '1')
    assert building.attributes['name'] == 'Test Building'
    assert building.attributes['height'] == '50'

    road = next(r for r in rs if r.uid == '2')
    assert road.attributes['name'] == 'Main Street'
    assert road.attributes['type'] == 'residential'


def test_parse_multiple_geometries_last_wins():
    """Test that when multiple geometries exist, the last one is used."""
    xml = """
    <wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs"
                            xmlns:gml="http://www.opengis.net/gml"
                            xmlns:app="http://example.com/app">
        <wfs:member>
            <app:feature gml:id="test.1">
                <app:name>Multi Geometry Feature</app:name>
                <gml:Point srsName="EPSG:4326">
                    <gml:coordinates>-71.1,42.2</gml:coordinates>
                </gml:Point>
                <gml:LineString srsName="EPSG:4326">
                    <gml:coordinates>-71.1,42.2 -71.09,42.21</gml:coordinates>
                </gml:LineString>
            </app:feature>
        </wfs:member>
        <wfs:member>
            <app:feature gml:id="test.2">
                <app:name>Single Geometry Feature</app:name>
                <gml:Point srsName="EPSG:4326">
                    <gml:coordinates>-70.1,41.2</gml:coordinates>
                </gml:Point>
            </app:feature>
        </wfs:member>
    </wfs:FeatureCollection>
    """

    rs = featureinfo.parse(xml)

    assert len(rs) == 2

    multi_geom = next(r for r in rs if r.uid == 'test.1')
    assert multi_geom.attributes['name'] == 'Multi Geometry Feature'
    assert multi_geom.shape is not None

    single_geom = next(r for r in rs if r.uid == 'test.2')
    assert single_geom.attributes['name'] == 'Single Geometry Feature'
    assert single_geom.shape is not None


def test_real_life_examples():
    import os
    for de in os.scandir(os.path.dirname(__file__) + '/featureinfo'):
        xml = gws.u.read_file(de.path)
        rs = featureinfo.parse(xml, default_crs=gws.lib.crs.WGS84)
        assert isinstance(rs, list)
        # for r in rs:
        #     print(f'\n{de.name} {r.uid=} {r.attributes=}')
