"""Tests for the namespace module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx


def test_get():
    ns = xmlx.namespace.get('GML')
    assert ns and ns.uid == 'GML' and ns.xmlns == 'gml' and ns.uri == 'http://www.opengis.net/gml/3.2'

    ns = xmlx.namespace.get('GML_2')
    assert ns and ns.uid == 'GML_2' and ns.xmlns == 'gml' and ns.uri == 'http://www.opengis.net/gml'

    ns = xmlx.namespace.get('OWS_11')
    assert ns and ns.xmlns == 'ows' and ns.uri == 'http://www.opengis.net/ows/1.1'

    assert xmlx.namespace.get('nonexistent') is None
    assert xmlx.namespace.get('gml') is None


def test_require():
    assert xmlx.namespace.require('WFS').uri == 'http://www.opengis.net/wfs/2.0'
    with u.raises(xmlx.NamespaceError):
        xmlx.namespace.require('nonexistent')


def test_constants():
    assert xmlx.namespace.ns.WFS is xmlx.namespace.require('WFS')
    with u.raises(AttributeError):
        xmlx.namespace.ns.nonexistent


def test_find_by_uri():
    assert xmlx.namespace.find_by_uri('http://www.opengis.net/gml/3.2').uid == 'GML'
    assert xmlx.namespace.find_by_uri('http://www.opengis.net/gml').uid == 'GML_2'
    assert xmlx.namespace.find_by_uri('urn:nonexistent') is None


def test_find_by_xmlns():
    assert xmlx.namespace.find_by_xmlns('wfs').uid == 'WFS'
    assert xmlx.namespace.find_by_xmlns('nonexistent') is None


def test_register():
    xmlx.namespace.unregister_all()

    ns_1 = xmlx.namespace.new('ns_1', 'urn:ns_1')
    xmlx.namespace.register(ns_1)
    assert xmlx.namespace.find_by_uri('urn:ns_1') is ns_1
    assert xmlx.namespace.find_by_xmlns('ns_1') is ns_1

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


def test_resolve_name():
    assert xmlx.namespace.resolve_name('foo') == 'foo'
    assert xmlx.namespace.resolve_name('{urn:x}foo') == '{urn:x}foo'
    assert xmlx.namespace.resolve_name('GML:foo') == '{http://www.opengis.net/gml/3.2}foo'
    with u.raises(xmlx.NamespaceError):
        xmlx.namespace.resolve_name('gml:foo')


def test_clark_name():
    ns = xmlx.namespace.ns.GML
    assert xmlx.namespace.clark_name('foo', ns) == '{http://www.opengis.net/gml/3.2}foo'
    assert xmlx.namespace.clark_name('foo', 'urn:x') == '{urn:x}foo'
    assert xmlx.namespace.clark_name('foo', None) == 'foo'
    assert xmlx.namespace.split_clark_name('{urn:x}foo') == ('urn:x', 'foo')
    assert xmlx.namespace.split_clark_name('foo') == ('', 'foo')


def test_split_name():
    assert xmlx.namespace.split_name('') == ('', '')
    assert xmlx.namespace.split_name('name') == ('', 'name')
    assert xmlx.namespace.split_name('somens:tag') == ('somens', 'tag')


def test_qualify_name():
    ns = xmlx.namespace.get('SOAP')
    assert xmlx.namespace.qualify_name('gml:foo', ns, replace=False) == 'gml:foo'
    assert xmlx.namespace.qualify_name('gml:foo', ns, replace=True) == 'soap:foo'
    assert xmlx.namespace.qualify_name('foo', ns) == 'soap:foo'
    assert xmlx.namespace.qualify_name('foo', None) == 'foo'


def test_unqualify_name():
    assert xmlx.namespace.unqualify_name('name') == 'name'
    assert xmlx.namespace.unqualify_name('gml:name') == 'name'
    assert xmlx.namespace.unqualify_name('{urn:x}name') == 'name'


def test_declarations_default():
    ns = xmlx.namespace.new('', 'http://www.w3.org/2003/05/soap-envelope')
    assert xmlx.namespace.declarations([ns]) == {'xmlns': 'http://www.w3.org/2003/05/soap-envelope'}


def test_declarations_element():
    ns_list = [
        xmlx.namespace.new('', 'http://www.w3.org/2003/05/soap-envelope'),
        xmlx.namespace.require('GML'),
    ]
    assert xmlx.namespace.declarations(ns_list) == {
        'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
        'xmlns:gml': 'http://www.opengis.net/gml/3.2',
    }


def test_declarations_custom_prefix():
    ns_list = [xmlx.namespace.require('GML')]
    assert xmlx.namespace.declarations(ns_list, {'http://www.opengis.net/gml/3.2': 'g'}) == {
        'xmlns:g': 'http://www.opengis.net/gml/3.2',
    }


def test_declarations_schema():
    ns_list = [
        xmlx.namespace.new('', 'http://www.w3.org/2003/05/soap-envelope', 'http://www.w3.org/2003/05/soap-envelope/'),
        xmlx.namespace.require('GML'),
    ]
    assert xmlx.namespace.declarations(ns_list, with_schema_locations=True) == {
        'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
        'xmlns:gml': 'http://www.opengis.net/gml/3.2',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': 'http://www.w3.org/2003/05/soap-envelope http://www.w3.org/2003/05/soap-envelope/ http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd',
    }
