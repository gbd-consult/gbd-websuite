import gws
import gws.types as t

import gws.lib.xmlx as xmlx


def test_iter():
    xml = '''
        <root>
            <a/>
            <b/>
            <c/>
        </root>
    '''
    root = xmlx.from_string(xml)
    tags = [e.tag for e in root]
    assert tags == ['a', 'b', 'c']


def test_find_with_namespaces():
    xml = '''
        <root 
                xmlns:gml="http://www.opengis.net/gml"
                xmlns:other="foobar"
        >
            <gml:a test="gml1"/>
            <a test="just"/>
            <other:a test="another"/>
            <gml:a test="gml2"/>
        </root>
    '''
    root = xmlx.from_string(xml)

    atts = [a.get('test') for a in root.findall('gml:a', {'gml': 'http://www.opengis.net/gml'})]
    assert atts == ['gml1', 'gml2']


def test_find_with_resolved_namespaces():
    xml = '''
        <root 
                xmlns:gml="http://www.opengis.net/gml"
                xmlns:gml3="http://www.opengis.net/gml/3.2"
                xmlns:other="foobar"
        >
            <gml:a test="gml1"/>
            <a test="just"/>
            <other:a test="another"/>
            <gml3:a test="gml2"/>
        </root>
    '''
    root = xmlx.from_string(xml, resolve_namespaces=True)

    atts = [a.get('test') for a in root.findall('{gml}a')]
    assert atts == ['gml1', 'gml2']

    atts = [a.get('test') for a in root.findall('{*}a')]
    assert atts == ['gml1', 'just', 'another', 'gml2']
