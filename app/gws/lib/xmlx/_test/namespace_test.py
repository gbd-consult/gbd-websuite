"""Tests for the namespace module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx


def test_get():
    ns = xmlx.namespace.get('gml')
    assert ns and ns.uid == 'gml' and ns.xmlns == 'gml' and ns.uri == 'http://www.opengis.net/gml/3.2'

    ns = xmlx.namespace.get('gml2')
    assert ns and ns.uid == 'gml2' and ns.xmlns == 'gml' and ns.uri == 'http://www.opengis.net/gml'

    ns = xmlx.namespace.get('ows11')
    assert ns and ns.xmlns == 'ows' and ns.uri == 'http://www.opengis.net/ows/1.1'

    assert xmlx.namespace.get('nonexistent') is None


def test_require():
    assert xmlx.namespace.require('wfs').uri == 'http://www.opengis.net/wfs/2.0'
    with u.raises(xmlx.NamespaceError):
        xmlx.namespace.require('nonexistent')


def test_split_name():
    assert xmlx.namespace.split_name('') == ('', '')
    assert xmlx.namespace.split_name('name') == ('', 'name')
    assert xmlx.namespace.split_name('somens:tag') == ('somens', 'tag')


def test_qualify_name():
    ns = xmlx.namespace.get('soap')
    assert xmlx.namespace.qualify_name('gml:foo', ns, replace=False) == 'gml:foo'
    assert xmlx.namespace.qualify_name('gml:foo', ns, replace=True) == 'soap:foo'
    assert xmlx.namespace.qualify_name('foo', ns) == 'soap:foo'
    assert xmlx.namespace.qualify_name('foo', None) == 'foo'


def test_unqualify_name():
    assert xmlx.namespace.unqualify_name('name') == 'name'
    assert xmlx.namespace.unqualify_name('gml:name') == 'name'


def test_declarations_default():
    ns = xmlx.namespace.require('soap')
    assert xmlx.namespace.declarations({'': ns}) == {'xmlns': 'http://www.w3.org/2003/05/soap-envelope'}


def test_declarations_element():
    ns_map = {
        '': xmlx.namespace.require('soap'),
        'gml': xmlx.namespace.require('gml'),
    }
    assert xmlx.namespace.declarations(ns_map) == {
        'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
        'xmlns:gml': 'http://www.opengis.net/gml/3.2',
    }


def test_declarations_schema():
    ns_map = {
        '': xmlx.namespace.require('soap'),
        'gml': xmlx.namespace.require('gml'),
    }
    assert xmlx.namespace.declarations(ns_map, with_schema_locations=True) == {
        'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
        'xmlns:gml': 'http://www.opengis.net/gml/3.2',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': 'http://www.w3.org/2003/05/soap-envelope http://www.w3.org/2003/05/soap-envelope/ http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd',
    }
