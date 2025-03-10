"""Tests for the namespace module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx


def test_find_by_uri():
    uri = 'http://www.opengis.net/gml'
    assert xmlx.namespace.find_by_uri(uri).__dict__ == {'uid': 'gml',
                                                        'xmlns': 'gml',
                                                        'uri': 'http://www.opengis.net/gml',
                                                        'schemaLocation': 'http://schemas.opengis.net/gml/3.2.1/gml.xsd',
                                                        'version': '3.2'}


def test_find_by_uri_versioned():
    uri = 'http://www.opengis.net/gml/3.2'
    assert xmlx.namespace.find_by_uri(uri).__dict__ == {'uid': 'gml',
                                                        'xmlns': 'gml',
                                                        'uri': 'http://www.opengis.net/gml',
                                                        'schemaLocation': 'http://schemas.opengis.net/gml/3.2.1/gml.xsd',
                                                        'version': '3.2'}


def test_find_by_uri_versioned2():
    uri = 'http://www.opengis.net/gml/9999'
    assert xmlx.namespace.find_by_uri(uri).__dict__ == {'uid': 'gml',
                                                        'xmlns': 'gml',
                                                        'uri': 'http://www.opengis.net/gml',
                                                        'schemaLocation': 'http://schemas.opengis.net/gml/3.2.1/gml.xsd',
                                                        'version': '3.2'}


def test_get():
    xmlns = 'gml'
    assert xmlx.namespace.get(xmlns).__dict__ == {'uid': 'gml',
                                                            'xmlns': 'gml',
                                                            'uri': 'http://www.opengis.net/gml',
                                                            'schemaLocation': 'http://schemas.opengis.net/gml/3.2.1/gml.xsd',
                                                            'version': '3.2'}


def test_get_uid():
    assert xmlx.namespace.get('csw').__dict__ == {'uid': 'csw',
                                                  'xmlns': 'csw',
                                                  'uri': 'http://www.opengis.net/cat/csw',
                                                  'schemaLocation': 'http://schemas.opengis.net/csw/2.0.2/csw.xsd',
                                                  'version': '2.0.2'}


def test_get_xmlns():
    assert xmlx.namespace.get('gml').__dict__ == {'schemaLocation': 'http://schemas.opengis.net/gml/3.2.1/gml.xsd',
                                                  'uid': 'gml',
                                                  'uri': 'http://www.opengis.net/gml',
                                                  'version': '3.2',
                                                  'xmlns': 'gml'}


def test_get_uri():
    assert xmlx.namespace.get('http://www.isotc211.org/2005/srv').__dict__ == {
        'schemaLocation': 'http://schemas.opengis.net/iso/19139/20070417/srv/1.0/srv.xsd',
        'uid': 'srv',
        'uri': 'http://www.isotc211.org/2005/srv',
        'version': '',
        'xmlns': 'srv'}


def test_register():
    ns = gws.XmlNamespace(uid='foo', xmlns='bar', uri='http://www.foobar.de')
    xmlx.namespace.register(ns)
    assert ns.uid in xmlx.namespace._INDEX.uid
    assert ns.xmlns in xmlx.namespace._INDEX.xmlns
    assert ns.uri in xmlx.namespace._INDEX.uri


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
    assert xmlx.namespace.extract(ns) == (xmlx.namespace.get('gml'), 'tag')


def test_extract_uri():
    uri = '{http://www.opengis.net/gml}tag'
    assert xmlx.namespace.extract(uri) == (xmlx.namespace.get('http://www.opengis.net/gml'), 'tag')


def test_extract_else():
    name = 'foo'
    assert xmlx.namespace.extract(name) == (None, 'foo')


# cant deal with clark notation
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


def test_unqualify_default_uri():
    name = '{http://www.opengis.net/gml}tag'
    d_ns = xmlx.namespace.find_by_uri('http://www.opengis.net/gml')
    assert xmlx.namespace.unqualify_default(name, d_ns) == 'tag'


def test_unqualify_default_xmlx():
    name = 'gml:tag'
    d_ns = xmlx.namespace.get('gml')
    assert xmlx.namespace.unqualify_default(name, d_ns) == 'tag'


def test_unqualify_default_else():
    name = 'gml:tag'
    d_ns = xmlx.namespace.get('soap')
    assert xmlx.namespace.unqualify_default(name, d_ns) == 'gml:tag'


def test_clarkify_name():
    name = 'gml:tag'
    assert xmlx.namespace.clarkify_name(name) == '{http://www.opengis.net/gml}tag'


def test_clarkify_name_else():
    name = 'tag'
    assert xmlx.namespace.clarkify_name(name) == 'tag'


def test_declarations_empty():
    assert xmlx.namespace.declarations() == {}


def test_declarations_default():
    d_ns = xmlx.namespace.get('soap')
    assert xmlx.namespace.declarations(d_ns) == {'xmlns': 'http://www.w3.org/2003/05/soap-envelope'}


def test_declarations_element():
    d_ns = xmlx.namespace.get('soap')
    xml_str = '''
                <root 
                        xmlns:gml="http://www.opengis.net/gml"
                >
                    <gml:a test="gml1"/>
                    <a test="just"/>
                    <a test="another"/>
                    <gml:a test="gml2"/>
                </root>
            '''
    ixmlelement = xmlx.parser.from_string(xml_str)
    assert xmlx.namespace.declarations(d_ns, ixmlelement) == {'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
                                                              'xmlns:gml': 'http://www.opengis.net/gml/3.2',}


def test_declarations_extra():
    d_ns = xmlx.namespace.get('soap')
    extra = [xmlx.namespace.get('gml')]
    assert xmlx.namespace.declarations(d_ns,extra_ns=extra) == {'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
                                                              'xmlns:gml': 'http://www.opengis.net/gml/3.2',}


def test_declarations_schema():
    d_ns = xmlx.namespace.get('soap')
    assert xmlx.namespace.declarations(d_ns, with_schema_locations=True) == { 'xmlns': 'http://www.w3.org/2003/05/soap-envelope',
                                                                              'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                                                                              'xsi:schemaLocation': 'http://www.w3.org/2003/05/soap-envelope http://https://www.w3.org/2003/05/soap-envelope/',}