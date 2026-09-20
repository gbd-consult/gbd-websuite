"""Tests for the namespace module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx


def test_find_well_known():
    ns = xmlx.namespace.find_well_known('GML')
    assert ns and ns.prefix == 'gml' and ns.uri == 'http://www.opengis.net/gml/3.2'

    ns = xmlx.namespace.find_well_known('GML_2')
    assert ns and ns.prefix == 'gml' and ns.uri == 'http://www.opengis.net/gml'

    ns = xmlx.namespace.find_well_known('OWS_11')
    assert ns and ns.prefix == 'ows' and ns.uri == 'http://www.opengis.net/ows/1.1'

    assert xmlx.namespace.find_well_known('nonexistent') is None
    assert xmlx.namespace.find_well_known('gml') is None


def test_constants():
    assert xmlx.namespace.c.WFS is xmlx.namespace.find_well_known('WFS')
    assert xmlx.namespace.c.WFS.uri == 'http://www.opengis.net/wfs/2.0'


def test_find_by_uri():
    assert xmlx.namespace.find_by_uri('http://www.opengis.net/gml/3.2') is xmlx.namespace.c.GML
    assert xmlx.namespace.find_by_uri('http://www.opengis.net/gml') is xmlx.namespace.c.GML_2
    assert xmlx.namespace.find_by_uri('urn:nonexistent') is None


def test_find_by_prefix():
    assert xmlx.namespace.find_by_prefix('wfs') is xmlx.namespace.c.WFS
    assert xmlx.namespace.find_by_prefix('nonexistent') is None


def test_register():
    xmlx.namespace.unregister_all()

    ns_1 = xmlx.namespace.new('ns_1', 'urn:ns_1')
    xmlx.namespace.register(ns_1)
    assert xmlx.namespace.find_by_uri('urn:ns_1') is ns_1
    assert xmlx.namespace.find_by_prefix('ns_1') is ns_1

    xmlx.namespace.register(xmlx.namespace.new('ns_1', 'urn:ns_1'))
    assert xmlx.namespace.find_by_uri('urn:ns_1') is ns_1

    with u.raises(xmlx.NamespaceError):
        xmlx.namespace.register(xmlx.namespace.new('ns_2', 'urn:ns_1'))
    with u.raises(xmlx.NamespaceError):
        xmlx.namespace.register(xmlx.namespace.new('ns_1', 'urn:ns_2'))
    with u.raises(xmlx.NamespaceError):
        xmlx.namespace.register(xmlx.namespace.new('gml', 'urn:ns_2'))

    xmlx.namespace.unregister_all()
    assert xmlx.namespace.find_by_uri('urn:ns_1') is None


def test_parse_name():
    assert xmlx.namespace.parse_name('') == ('', '', '')
    assert xmlx.namespace.parse_name('name') == ('', '', 'name')
    assert xmlx.namespace.parse_name('somens:tag') == ('', 'somens', 'tag')
    assert xmlx.namespace.parse_name('{urn:x}foo') == ('urn:x', '', 'foo')
    assert xmlx.namespace.parse_name('{urn:x:y}foo') == ('urn:x:y', '', 'foo')


def test_full_name():
    ns = xmlx.namespace.c.GML
    assert xmlx.namespace.full_name('foo', ns) == '{http://www.opengis.net/gml/3.2}foo'
    assert xmlx.namespace.full_name('p:foo', ns) == '{http://www.opengis.net/gml/3.2}foo'
    assert xmlx.namespace.full_name('{urn:y}foo', ns) == '{http://www.opengis.net/gml/3.2}foo'
    assert xmlx.namespace.full_name('foo', 'urn:x') == '{urn:x}foo'
    assert xmlx.namespace.full_name('foo', None) == 'foo'
    assert xmlx.namespace.full_name('p:foo', None) == 'foo'


def test_plain_name():
    assert xmlx.namespace.plain_name('name') == 'name'
    assert xmlx.namespace.plain_name('gml:name') == 'name'
    assert xmlx.namespace.plain_name('{urn:x}name') == 'name'


def test_declarations_default():
    ns = xmlx.namespace.new('', 'http://www.w3.org/2003/05/soap-envelope')
    assert xmlx.namespace.declarations([ns]) == {'xmlns': 'http://www.w3.org/2003/05/soap-envelope'}


def test_declarations_element():
    ns_list = [
        xmlx.namespace.new('', 'http://www.w3.org/2003/05/soap-envelope'),
        xmlx.namespace.c.GML,
    ]
    assert xmlx.namespace.declarations(ns_list) == {
        'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
        'xmlns:gml': 'http://www.opengis.net/gml/3.2',
    }


def test_declarations_custom_prefix():
    ns_list = [xmlx.namespace.c.GML]
    assert xmlx.namespace.declarations(ns_list, {'http://www.opengis.net/gml/3.2': 'g'}) == {
        'xmlns:g': 'http://www.opengis.net/gml/3.2',
    }


def test_declarations_schema():
    ns_list = [
        xmlx.namespace.new('', 'http://www.w3.org/2003/05/soap-envelope', 'http://www.w3.org/2003/05/soap-envelope/'),
        xmlx.namespace.c.GML,
    ]
    assert xmlx.namespace.declarations(ns_list, with_schema_locations=True) == {
        'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
        'xmlns:gml': 'http://www.opengis.net/gml/3.2',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': 'http://www.w3.org/2003/05/soap-envelope http://www.w3.org/2003/05/soap-envelope/ http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd',
    }
