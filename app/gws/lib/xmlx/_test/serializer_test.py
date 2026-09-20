"""Tests for the serializer module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx
from gws.lib.xmlx import tag


def _ns(uid, uri, xmlns=None, schema=''):
    return gws.XmlNamespace(uid=uid, xmlns=xmlns or uid, uri=uri, schemaLocation=schema, extendsGml=False)


def test_to_string_basic():
    el = tag('a', {'x': 1, 'y': None}, 'text', tag('b'), 'tail')
    assert el.to_string() == '<a x="1">text<b/>tail</a>'


def test_to_string_escaping():
    el = tag('a', {'x': 'a<b>"c"&\n'}, '<b>&"')
    assert el.to_string() == '<a x="a&lt;b&gt;&quot;c&quot;&amp;&#xa;">&lt;b&gt;&amp;"</a>'


def test_to_string_values():
    el = tag('a', {'b': True, 'f': 1.5, 'i': 0}, False)
    assert el.to_string() == '<a b="true" f="1.5" i="0">false</a>'


def test_to_string_compact():
    el = tag('a', '  x   y  ', tag('b', ' z '))
    assert el.to_string(gws.XmlOptions(compactWhitespace=True)) == '<a>x y<b>z</b></a>'


def test_to_string_declaration_and_doctype():
    el = tag('a')
    assert el.to_string(gws.XmlOptions(withXmlDeclaration=True)) == '<?xml version="1.0" encoding="UTF-8"?><a/>'
    assert el.to_string(gws.XmlOptions(doctype='a SYSTEM "a.dtd"')) == '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE a SYSTEM "a.dtd"><a/>'


def test_to_string_with_namespaces():
    aaa_ns = _ns('aaa', 'http://aaa')
    bbb_ns = _ns('bbb', 'http://bbb')

    el = tag(
        'aaa:a/bbb:b',
        {
            'a1': 'A1',
            'aaa:a2': 'A2',
        },
        tag('bbb:sub'),
    )

    opts = gws.XmlOptions(
        namespaces={
            'aaa': aaa_ns,
            'bbb': bbb_ns,
        },
        withNamespaceDeclarations=True,
    )
    xml = el.to_string(opts)
    u.check.xml(xml, """
        <aaa:a xmlns:aaa="http://aaa" xmlns:bbb="http://bbb">
            <bbb:b a1="A1" aaa:a2="A2">
                <bbb:sub/>
            </bbb:b>
        </aaa:a>
    """)


def test_to_string_without_declarations():
    el = tag('aaa:a', tag('bbb:b'))
    opts = gws.XmlOptions(namespaces={'aaa': _ns('aaa', 'http://aaa'), 'bbb': _ns('bbb', 'http://bbb')})
    assert el.to_string(opts) == '<aaa:a><bbb:b/></aaa:a>'


def test_to_string_with_unknown_namespace():
    el = tag('aaa:a/bbb:b')
    with u.raises(xmlx.NamespaceError):
        el.to_string()


def test_to_string_with_well_known_namespaces():
    el = tag('wfs:a', {'xlink:href': 'x'}, tag('gml:b'))
    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))
    u.check.xml(xml, """
        <wfs:a
            xlink:href="x"
            xmlns:gml="http://www.opengis.net/gml/3.2"
            xmlns:wfs="http://www.opengis.net/wfs/2.0"
            xmlns:xlink="http://www.w3.org/1999/xlink">
            <gml:b/>
        </wfs:a>
    """)


def test_to_string_explicit_map_overrides_well_known():
    el = tag('wfs:a', tag('gml:b'))
    xml = el.to_string(gws.XmlOptions(namespaces={'gml': xmlx.namespace.require('gml2')}, withNamespaceDeclarations=True))
    u.check.xml(xml, """
        <wfs:a xmlns:gml="http://www.opengis.net/gml" xmlns:wfs="http://www.opengis.net/wfs/2.0">
            <gml:b/>
        </wfs:a>
    """)


def test_to_string_with_schema_locations():
    el = tag('wfs:a')
    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True, withSchemaLocations=True))
    u.check.xml(xml, """
        <wfs:a
            xmlns:wfs="http://www.opengis.net/wfs/2.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd"/>
    """)


def test_to_string_with_default_namespace():
    aaa_ns = _ns('aaa', 'http://aaa')
    bbb_ns = _ns('bbb', 'http://bbb')

    el = tag(
        'aaa:a/bbb:b',
        {
            'a1': 'A1',
            'bbb:a2': 'A2',
        },
        tag('bbb:sub'),
        tag('aaa:sub'),
    )

    opts = gws.XmlOptions(
        namespaces={
            'aaa': aaa_ns,
            'bbb': bbb_ns,
        },
        defaultNamespace=aaa_ns,
        withNamespaceDeclarations=True,
    )
    xml = el.to_string(opts)
    u.check.xml(xml, """
        <a xmlns="http://aaa" xmlns:bbb="http://bbb">
            <bbb:b a1="A1" bbb:a2="A2">
                <bbb:sub/>
                <sub/>
            </bbb:b>
        </a>
    """)


def test_attributes_in_default_namespace_keep_prefix():
    aaa_ns = _ns('aaa', 'http://aaa')
    el = tag('aaa:a', {'aaa:x': '1'})
    opts = gws.XmlOptions(namespaces={'aaa': aaa_ns}, defaultNamespace=aaa_ns, withNamespaceDeclarations=True)
    u.check.xml(el.to_string(opts), '<a aaa:x="1" xmlns="http://aaa" xmlns:aaa="http://aaa"/>')


def test_xml_prefix_is_never_declared():
    el = tag('a', {'xml:lang': 'de'})
    assert el.to_string(gws.XmlOptions(withNamespaceDeclarations=True)) == '<a xml:lang="de"/>'


def test_custom_prefixes():
    el = tag('wfs:a', tag('gml:b', {'gml:id': 'x'}))
    opts = gws.XmlOptions(
        customXmlns={'http://www.opengis.net/gml/3.2': 'g', 'http://www.opengis.net/wfs/2.0': 'w'},
        withNamespaceDeclarations=True,
    )
    u.check.xml(el.to_string(opts), """
        <w:a xmlns:g="http://www.opengis.net/gml/3.2" xmlns:w="http://www.opengis.net/wfs/2.0">
            <g:b g:id="x"/>
        </w:a>
    """)


def test_element_namespaces():
    gml = xmlx.namespace.require('gml')
    el = tag('a', tag('gml:Point', tag('gml:pos', '1 2')))
    el[0].namespaces['gml'] = gml
    u.check.xml(el.to_string(), """
        <a>
            <gml:Point xmlns:gml="http://www.opengis.net/gml/3.2">
                <gml:pos>1 2</gml:pos>
            </gml:Point>
        </a>
    """)


def test_element_namespaces_resolve_unknown_prefixes():
    el = tag('a', tag('foo:b', {'foo:x': 1}, tag('foo:c')))
    el[0].namespaces['foo'] = _ns('foo', 'http://foo')
    u.check.xml(el.to_string(), '<a><foo:b foo:x="1" xmlns:foo="http://foo"><foo:c/></foo:b></a>')


def test_element_namespaces_with_custom_prefix():
    gml = xmlx.namespace.require('gml')
    el = tag('a', tag('gml:Point'))
    el[0].namespaces['gml'] = gml
    opts = gws.XmlOptions(customXmlns={gml.uri: 'g'})
    u.check.xml(el.to_string(opts), '<a><g:Point xmlns:g="http://www.opengis.net/gml/3.2"/></a>')


def test_element_namespaces_and_root_declarations():
    gml = xmlx.namespace.require('gml')
    el = tag('wfs:a', tag('gml:Point'))
    el[0].namespaces['gml'] = gml
    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))
    u.check.xml(xml, """
        <wfs:a xmlns:wfs="http://www.opengis.net/wfs/2.0">
            <gml:Point xmlns:gml="http://www.opengis.net/gml/3.2"/>
        </wfs:a>
    """)


def test_parsed_declarations_roundtrip():
    doc = xmlx.from_string('<a xmlns="http://d" xmlns:x="http://x"><b/></a>')
    assert doc.to_string() == '<a xmlns="http://d" xmlns:x="http://x"><b/></a>'


def test_invalid_element_name():
    with u.raises(xmlx.WriteError):
        tag('a', tag('my col')).to_string()
    with u.raises(xmlx.WriteError):
        tag('1st').to_string()
    with u.raises(xmlx.WriteError):
        tag('a<b').to_string()


def test_invalid_attribute_name():
    with u.raises(xmlx.WriteError):
        tag('a', {'my col': 1}).to_string()
    with u.raises(xmlx.WriteError):
        tag('a', {'x"y': 1}).to_string()


def test_invalid_custom_prefix():
    el = tag('wfs:a')
    with u.raises(xmlx.WriteError):
        el.to_string(gws.XmlOptions(customXmlns={'http://www.opengis.net/wfs/2.0': 'bad prefix'}))


def test_non_ascii_names():
    assert tag('Straße', {'größe': 1}, tag('日本')).to_string() == '<Straße größe="1"><日本/></Straße>'
