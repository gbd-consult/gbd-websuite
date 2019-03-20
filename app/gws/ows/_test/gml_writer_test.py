import gws.ows.gml
import gws.gis.shape
import gws.tools.xml3

"""
drop table if exists tt;
create temporary table tt (w text);

insert into tt values
('POINT (30 10)'),
('LINESTRING (30 10, 10 30, 40 40)'),
('POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))'),
('POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10),(20 30, 35 35, 30 20, 20 30))'),
('MULTIPOINT ((10 40), (40 30), (20 20), (30 10))'),
('MULTIPOINT (10 40, 40 30, 20 20, 30 10)'),
('MULTILINESTRING ((10 10, 20 20, 10 40),(40 40, 30 30, 40 20, 30 10))'),
('MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)),((15 5, 40 10, 10 20, 5 10, 15 5)))'),
('MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)),((20 35, 10 30, 10 10, 30 5, 45 20, 20 35),(30 20, 20 15, 20 25, 30 20)))'
)

select w, st_asgml(3, st_geomfromewkt('SRID=25832;'||w), 1, 4+1) as g from tt

"""

test = [
    {
        "w": "POINT (30 10)",
        "g": "<gml:Point srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:pos srsDimension=\"2\">30 10</gml:pos></gml:Point>"
    },
    {
        "w": "LINESTRING (30 10, 10 30, 40 40)",
        "g": "<gml:LineString srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:posList srsDimension=\"2\">30 10 10 30 40 40</gml:posList></gml:LineString>"
    },
    {
        "w": "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
        "g": "<gml:Polygon srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:exterior><gml:LinearRing><gml:posList srsDimension=\"2\">30 10 40 40 20 40 10 20 30 10</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon>"
    },
    {
        "w": "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10),(20 30, 35 35, 30 20, 20 30))",
        "g": "<gml:Polygon srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:exterior><gml:LinearRing><gml:posList srsDimension=\"2\">35 10 45 45 15 40 10 20 35 10</gml:posList></gml:LinearRing></gml:exterior><gml:interior><gml:LinearRing><gml:posList srsDimension=\"2\">20 30 35 35 30 20 20 30</gml:posList></gml:LinearRing></gml:interior></gml:Polygon>"
    },
    {
        "w": "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))",
        "g": "<gml:MultiPoint srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:pointMember><gml:Point><gml:pos srsDimension=\"2\">10 40</gml:pos></gml:Point></gml:pointMember><gml:pointMember><gml:Point><gml:pos srsDimension=\"2\">40 30</gml:pos></gml:Point></gml:pointMember><gml:pointMember><gml:Point><gml:pos srsDimension=\"2\">20 20</gml:pos></gml:Point></gml:pointMember><gml:pointMember><gml:Point><gml:pos srsDimension=\"2\">30 10</gml:pos></gml:Point></gml:pointMember></gml:MultiPoint>"
    },
    {
        "w": "MULTIPOINT (10 40, 40 30, 20 20, 30 10)",
        "g": "<gml:MultiPoint srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:pointMember><gml:Point><gml:pos srsDimension=\"2\">10 40</gml:pos></gml:Point></gml:pointMember><gml:pointMember><gml:Point><gml:pos srsDimension=\"2\">40 30</gml:pos></gml:Point></gml:pointMember><gml:pointMember><gml:Point><gml:pos srsDimension=\"2\">20 20</gml:pos></gml:Point></gml:pointMember><gml:pointMember><gml:Point><gml:pos srsDimension=\"2\">30 10</gml:pos></gml:Point></gml:pointMember></gml:MultiPoint>"
    },
    {
        "w": "MULTILINESTRING ((10 10, 20 20, 10 40),(40 40, 30 30, 40 20, 30 10))",
        "g": "<gml:MultiCurve srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:curveMember><gml:LineString><gml:posList srsDimension=\"2\">10 10 20 20 10 40</gml:posList></gml:LineString></gml:curveMember><gml:curveMember><gml:LineString><gml:posList srsDimension=\"2\">40 40 30 30 40 20 30 10</gml:posList></gml:LineString></gml:curveMember></gml:MultiCurve>"
    },
    {
        "w": "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)),((15 5, 40 10, 10 20, 5 10, 15 5)))",
        "g": "<gml:MultiSurface srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension=\"2\">30 20 45 40 10 40 30 20</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember><gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension=\"2\">15 5 40 10 10 20 5 10 15 5</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
    },
    {
        "w": "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)),((20 35, 10 30, 10 10, 30 5, 45 20, 20 35),(30 20, 20 15, 20 25, 30 20)))",
        "g": "<gml:MultiSurface srsName=\"urn:ogc:def:crs:EPSG::25832\"><gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension=\"2\">40 40 20 45 45 30 40 40</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember><gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension=\"2\">20 35 10 30 10 10 30 5 45 20 20 35</gml:posList></gml:LinearRing></gml:exterior><gml:interior><gml:LinearRing><gml:posList srsDimension=\"2\">30 20 20 15 20 25 30 20</gml:posList></gml:LinearRing></gml:interior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
    }
]

def test_gml_writer():
    for t in test:
        s = gws.gis.shape.from_wkt(t['w'], 'EPSG:25832')
        r = gws.ows.gml.shape_to_tag(s)
        g = gws.tools.xml3._string(r)
        assert t['g'] == g
