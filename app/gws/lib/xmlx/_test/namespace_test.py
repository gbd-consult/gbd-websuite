"""Tests for the namespace module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx


def test_find_by_uri():
    uri = 'http://www.opengis.net/gml'
    ns = xmlx.namespace.find_by_uri(uri)
    assert ns and ns.uid == 'gml2'


def test_get():
    xmlns = 'gml'
    ns = xmlx.namespace.get(xmlns)
    assert ns and ns.uid == 'gml'


# def test_register():
#     ns = gws.XmlNamespace(uid='foo', xmlns='bar', uri='http://www.foobar.de')
#     xmlx.namespace.register(ns)
#     assert ns.uid in xmlx.namespace._INDEX.uid
#     assert ns.xmlns in xmlx.namespace._INDEX.xmlns
#     assert ns.uri in xmlx.namespace._INDEX.uri


def test_split_name_empty():
    name = ''
    assert xmlx.namespace.split_name(name) == ('', '', '')


def test_split_name():
    name = '{http://example.com/namespace}element'
    assert xmlx.namespace.split_name(name) == ('', 'http://example.com/namespace', 'element')


def test_split_name_colon():
    name = 'somens:tag'
    assert xmlx.namespace.split_name(name) == ('somens', '', 'tag')


def test_split_name_else():
    name = 'name'
    assert xmlx.namespace.split_name(name) == ('', '', 'name')


def test_extract_xmlns():
    ns = 'gml:tag'
    assert xmlx.namespace.extract(ns) == (xmlx.namespace.find_by_xmlns('gml'), 'tag')


def test_extract_uri():
    uri = '{http://www.opengis.net/gml}tag'
    assert xmlx.namespace.extract(uri) == (xmlx.namespace.find_by_uri('http://www.opengis.net/gml'), 'tag')


def test_extract_else():
    name = 'foo'
    assert xmlx.namespace.extract(name) == (None, 'foo')


def test_qualify_name():
    name = 'gml:foo'
    ns = xmlx.namespace.get('soap')
    assert xmlx.namespace.qualify_name(name, ns, replace=False) == 'gml:foo'


def test_qualify_name_replace():
    name = 'gml:foo'
    ns = xmlx.namespace.get('soap')
    assert xmlx.namespace.qualify_name(name, ns, replace=True) == 'soap:foo'


def test_qualify_name_else():
    name = 'foo'
    ns = None
    assert xmlx.namespace.qualify_name(name, ns) == 'foo'


def test_unqualify_name():
    assert xmlx.namespace.unqualify_name('name') == 'name'


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
