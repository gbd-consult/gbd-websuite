"""Tests for the serializer module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx
from gws.lib.xmlx import tag


def _ns(xmlns, uri, schema=''):
    return xmlx.namespace.new(xmlns, uri, schema)


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
    el = tag(
        '{http://aaa}a/{http://bbb}b',
        {
            'a1': 'A1',
            '{http://aaa}a2': 'A2',
        },
        tag('{http://bbb}sub'),
    )
    el.namespaces = [_ns('aaa', 'http://aaa'), _ns('bbb', 'http://bbb')]

    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))
    u.check.xml(xml, """
        <aaa:a xmlns:aaa="http://aaa" xmlns:bbb="http://bbb">
            <bbb:b a1="A1" aaa:a2="A2">
                <bbb:sub/>
            </bbb:b>
        </aaa:a>
    """)


def test_to_string_without_declarations():
    el = tag('WFS:a', tag('GML:b'))
    assert el.to_string() == '<wfs:a><gml:b/></wfs:a>'


def test_to_string_with_unknown_namespace():
    el = tag('{http://aaa}a/{http://bbb}b')
    with u.raises(xmlx.NamespaceError):
        el.to_string()
    with u.raises(xmlx.NamespaceError):
        el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))


def test_prefixed_name_is_rejected():
    el = tag('a')
    el.tag = 'x:a'
    with u.raises(xmlx.WriteError):
        el.to_string()
    el = tag('a')
    el.attrib['x:b'] = 1
    with u.raises(xmlx.WriteError):
        el.to_string()


def test_to_string_with_well_known_namespaces():
    el = tag('WFS:a', {'XLINK:href': 'x'}, tag('GML:b'))
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


def test_to_string_with_registered_namespace():
    xmlx.namespace.unregister_all()
    xmlx.namespace.register(_ns('ns_1', 'http://ns_1'))
    el = tag('WFS:a', tag('{http://ns_1}b'))
    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))
    xmlx.namespace.unregister_all()
    u.check.xml(xml, """
        <wfs:a xmlns:ns_1="http://ns_1" xmlns:wfs="http://www.opengis.net/wfs/2.0">
            <ns_1:b/>
        </wfs:a>
    """)


def test_to_string_with_schema_locations():
    el = tag('WFS:a')
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
        '{http://aaa}a/{http://bbb}b',
        {
            'a1': 'A1',
            '{http://bbb}a2': 'A2',
        },
        tag('{http://bbb}sub'),
        tag('{http://aaa}sub'),
        tag('bare'),
    )
    el.namespaces = [bbb_ns]

    opts = gws.XmlOptions(
        defaultNamespace=aaa_ns,
        withNamespaceDeclarations=True,
    )
    xml = el.to_string(opts)
    u.check.xml(xml, """
        <a xmlns="http://aaa" xmlns:bbb="http://bbb">
            <bbb:b a1="A1" bbb:a2="A2">
                <bbb:sub/>
                <sub/>
                <bare/>
            </bbb:b>
        </a>
    """)


def test_default_namespace_is_declared_with_schema_location():
    el = tag('WFS:a')
    xml = el.to_string(gws.XmlOptions(defaultNamespace=xmlx.namespace.ns.WFS, withNamespaceDeclarations=True, withSchemaLocations=True))
    u.check.xml(xml, """
        <a xmlns="http://www.opengis.net/wfs/2.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd"/>
    """)


def test_attributes_in_default_namespace_keep_prefix():
    el = tag('WFS:a', {'WFS:x': '1'})
    opts = gws.XmlOptions(defaultNamespace=xmlx.namespace.ns.WFS, withNamespaceDeclarations=True)
    u.check.xml(el.to_string(opts), '<a wfs:x="1" xmlns="http://www.opengis.net/wfs/2.0" xmlns:wfs="http://www.opengis.net/wfs/2.0"/>')


def test_xml_prefix_is_never_declared():
    el = tag('a', {'XML:lang': 'de'})
    assert el.to_string(gws.XmlOptions(withNamespaceDeclarations=True)) == '<a xml:lang="de"/>'


def test_adhoc_prefix_is_never_declared():
    el = tag('a', tag('{adhoc:foo}b', {'{adhoc:foo}x': 1}))
    assert el.to_string(gws.XmlOptions(withNamespaceDeclarations=True)) == '<a><foo:b foo:x="1"/></a>'


def test_custom_prefixes():
    el = tag('WFS:a', tag('GML:b', {'GML:id': 'x'}))
    opts = gws.XmlOptions(
        customXmlns={'http://www.opengis.net/gml/3.2': 'g', 'http://www.opengis.net/wfs/2.0': 'w'},
        withNamespaceDeclarations=True,
    )
    u.check.xml(el.to_string(opts), """
        <w:a xmlns:g="http://www.opengis.net/gml/3.2" xmlns:w="http://www.opengis.net/wfs/2.0">
            <g:b g:id="x"/>
        </w:a>
    """)


def test_prefix_collision():
    el = tag('GML:a', tag('GML_2:b'))
    with u.raises(xmlx.NamespaceError):
        el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))


def test_element_namespaces():
    gml = xmlx.namespace.ns.GML
    el = tag('a', tag('GML:Point', tag('GML:pos', '1 2')))
    el[0].namespaces.append(gml)
    u.check.xml(el.to_string(), """
        <a>
            <gml:Point xmlns:gml="http://www.opengis.net/gml/3.2">
                <gml:pos>1 2</gml:pos>
            </gml:Point>
        </a>
    """)


def test_element_namespaces_resolve_unknown_uris():
    el = tag('a', tag('{http://foo}b', {'{http://foo}x': 1}, tag('{http://foo}c')))
    el[0].namespaces.append(_ns('foo', 'http://foo'))
    u.check.xml(el.to_string(), '<a><foo:b foo:x="1" xmlns:foo="http://foo"><foo:c/></foo:b></a>')


def test_element_namespaces_override_table_prefix():
    el = tag('a', tag('GML:b', tag('GML:c')))
    el[0].namespaces.append(_ns('g', 'http://www.opengis.net/gml/3.2'))
    u.check.xml(el.to_string(), '<a><g:b xmlns:g="http://www.opengis.net/gml/3.2"><g:c/></g:b></a>')


def test_element_namespaces_with_custom_prefix():
    gml = xmlx.namespace.ns.GML
    el = tag('a', tag('GML:Point'))
    el[0].namespaces.append(gml)
    opts = gws.XmlOptions(customXmlns={gml.uri: 'g'})
    u.check.xml(el.to_string(opts), '<a><g:Point xmlns:g="http://www.opengis.net/gml/3.2"/></a>')


def test_element_namespaces_and_root_declarations():
    gml = xmlx.namespace.ns.GML
    el = tag('WFS:a', tag('GML:Point'))
    el[0].namespaces.append(gml)
    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))
    u.check.xml(xml, """
        <wfs:a xmlns:wfs="http://www.opengis.net/wfs/2.0">
            <gml:Point xmlns:gml="http://www.opengis.net/gml/3.2"/>
        </wfs:a>
    """)


def test_unknown_uri_declared_on_child_is_not_declared_on_root():
    el = tag('a', tag('{http://foo}b'))
    el[0].namespaces.append(_ns('foo', 'http://foo'))
    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))
    u.check.xml(xml, '<a><foo:b xmlns:foo="http://foo"/></a>')


def test_serializing_does_not_mutate_the_tree():
    el = tag('WFS:a', tag('GML:b'))
    el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))
    assert el.namespaces == []


def test_parsed_declarations_roundtrip():
    doc = xmlx.from_string('<a xmlns="http://d" xmlns:x="http://x"><b/><x:c x:y="1"/></a>')
    assert doc.to_string() == '<a xmlns="http://d" xmlns:x="http://x"><b/><x:c x:y="1"/></a>'
    assert doc.to_string(gws.XmlOptions(withNamespaceDeclarations=True)) == '<a xmlns="http://d" xmlns:x="http://x"><b/><x:c x:y="1"/></a>'


def test_parsed_undeclared_prefix_roundtrip():
    doc = xmlx.from_string('<a><gml:b gml:x="1"/></a>')
    assert doc.to_string() == '<a><gml:b gml:x="1"/></a>'


def test_parsed_document_with_custom_prefix():
    doc = xmlx.from_string('<a xmlns:x="http://x"><x:c/></a>')
    assert doc.to_string(gws.XmlOptions(customXmlns={'http://x': 'y'})) == '<a xmlns:y="http://x"><y:c/></a>'


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
    el = tag('WFS:a')
    with u.raises(xmlx.WriteError):
        el.to_string(gws.XmlOptions(customXmlns={'http://www.opengis.net/wfs/2.0': 'bad prefix'}))


def test_non_ascii_names():
    assert tag('Straße', {'größe': 1}, tag('日本')).to_string() == '<Straße größe="1"><日本/></Straße>'
