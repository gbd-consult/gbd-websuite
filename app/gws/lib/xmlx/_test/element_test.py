"""Tests for the element module"""

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


def test_to_dict():
    xml = '''
        <root>
            <a>
                <b/>
            </a>
        </root>
        '''

    d = {'attrib': {},
         'children': ([{

             'attrib': {},
             'children': ([{

                 'attrib': {},
                 'children': [],
                 'tag': 'b',
                 'tail': '\n            ',
                 'text': ''}]),

             'tag': 'a',
             'tail': '\n        ',
             'text': '\n                '}]),

         'tag': 'root',
         'tail': '',
         'text': '\n            '
         }
    root = xmlx.from_string(xml)
    assert root.to_dict() == d


def test_to_string():
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
            <root 
                xmlns:gml="http://www.opengis.net/gml"
                xmlns:gml3="http://www.opengis.net/gml/3.2"
            >
            <gml:a test="gml1"/>
            <a test="just"/>
            <gml3:a test="gml2"/>
                <a>
                    <b/>
                </a>
            </root>'''
    xml_str = '''<root>
            <gml:a test="gml1"/>
            <a test="just"/>
            <gml:a test="gml2"/>
                <a>
                    <b/>
                </a>
            </root>'''
    root = xmlx.from_string(xml)
    assert root.to_string() == xml_str


def test_to_string_compact_whitespace():
    xml = '''   <root>
                    <a>
                        <b test="just"/>
                    </a>
                </root>'''
    root = xmlx.from_string(xml)
    assert root.to_string(compact_whitespace=True) == '<root><a><b test="just"/></a></root>'


def test_to_string_remove_namespaces():
    xml = '''<root 
                xmlns:gml="http://www.opengis.net/gml"
                xmlns:gml3="http://www.opengis.net/gml/3.2"
            >
            <gml:a test="gml1"/>
            <a test="just"/>
            <gml3:a test="gml2"/>
                <a>
                    <b/>
                </a>
            </root>'''

    xml_no_nspace = '''<root>
            <a test="gml1"/>
            <a test="just"/>
            <a test="gml2"/>
                <a>
                    <b/>
                </a>
            </root>'''
    root = xmlx.from_string(xml)
    assert root.to_string(remove_namespaces=True) == xml_no_nspace


def test_to_string_with_namespace_declarations():
    xml = '''<root 
                xmlns:gml="http://www.opengis.net/gml"
                xmlns:gml3="http://www.opengis.net/gml/3.2"
            >
            <gml:a test="gml1"/>
            <a test="just"/>
            <gml3:a test="gml2"/>
                <a>
                    <b/>
                </a>
            </root>'''

    xml_with_namespace_declarations = '''<root xmlns:gml="http://www.opengis.net/gml/3.2">
            <gml:a test="gml1"/>
            <a test="just"/>
            <gml:a test="gml2"/>
                <a>
                    <b/>
                </a>
            </root>'''

    root = xmlx.from_string(xml)
    assert root.to_string(with_namespace_declarations=True) == xml_with_namespace_declarations


def test_to_string_with_schema_locations():
    xml = '''<root
            xmlns:gml="http://www.opengis.net/gml"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	        xsi:schemaLocation="test"
	        >
            <gml:a test="gml1"/>
            <a test="just"/>
            <a test="gml2"/>
                <a>
                    <b/>
                </a>
            </root>'''

    xml_with_schema_locations = '''<root xsi:schemaLocation="test">
            <gml:a test="gml1"/>
            <a test="just"/>
            <a test="gml2"/>
                <a>
                    <b/>
                </a>
            </root>'''
    root = xmlx.from_string(xml)
    assert root.to_string(with_schema_locations=True) == xml_with_schema_locations


def test_to_string_with_xml_declaration():
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
               <root>
                    <a>
                        <b/>
                    </a>
                </root>'''

    xml_with_xml_declaration = '''<?xml version="1.0" encoding="UTF-8"?><root>
                    <a>
                        <b/>
                    </a>
                </root>'''

    root = xmlx.from_string(xml)
    assert root.to_string(with_xml_declaration=True) == xml_with_xml_declaration


def test_add():
    xml = '''
            <Root>
            </Root>
            '''
    test = xmlx.parser.from_string(xml)
    test.add('a', {'foo': 'bar'})
    assert test.to_string(compact_whitespace=True) == '<Root><a foo="bar"/></Root>'
    assert test.children()[0].to_string(compact_whitespace=True) == '<a foo="bar"/>'


def test_attr():
    xml = '''
                <Root attr= "val">
                    <a bar="foo"/>
                    <b test ="attr"/>
                    <c><deep>test2="attr2"</deep></c>
                </Root>
                '''
    test = xmlx.parser.from_string(xml)
    assert test.attr('attr') == 'val'
    assert test.attr('attr2') == None


def test_children():
    xml = '''
        <root>
            <a>
                <b/>
            </a>
            <c/>
        </root>
    '''
    test = xmlx.parser.from_string(xml)
    assert test.children()[0].to_string(compact_whitespace=True) == '<a><b/></a>'
    assert test.children()[1].to_string(compact_whitespace=True) == '<c/>'
    assert test.children()[0].children()[0].to_string(compact_whitespace=True) == '<b/>'


def test_findfirst():
    xml = '''
        <root a1="A1" a2="A2">
        text
            <nested>
                <deep>text2</deep>
                text3
                <single b="B1"/>
            </nested>
        </root>
    '''
    test = xmlx.parser.from_string(xml)
    assert test.findfirst().__dict__ == {'caseInsensitive': False, 'lname': 'nested', 'name': 'nested'}


def test_textof():
    xml = '''
                    <container>
                        text
                        <tag1><deep>xxx</deep></tag1>
                        <tag2>yyy</tag2>
                        <tag3>zzz</tag3>
                        <tag4></tag4>
                    </container>'''
    test = xmlx.parser.from_string(xml)
    assert not test.textof()
    assert test.textof('tag1/deep') == 'xxx'
    assert test.textof('tag2') == 'yyy'
    assert not test.textof('tag4')


def test_textlist():
    xml = '''
                <container>
                    <tag1> xxx </tag1>
                    <tag2> yyy </tag2>
                    <tag3> zzz </tag3>
                </container>'''
    test = xmlx.parser.from_string(xml)
    assert test.textlist() == ["xxx", "yyy", "zzz"]


def test_textlist_nested():
    xml = '''
            <root a1="A1" a2="A2">
            text
                <nested><deep>text3</deep></nested>
            </root>
        '''
    test = xmlx.parser.from_string(xml)
    assert test.textlist('nested/deep') == ['text3']


def test_textlist_empty():
    xml = '''
            <root>
            </root>
        '''
    test = xmlx.parser.from_string(xml)
    assert test.textlist() == []


def test_textlist_deep():
    xml = '''
                    <root a1="A1" a2="A2">
                    text
                        <nested>text1<deep>text2</deep></nested>
                        <nested2>text3<deep2>text4</deep2></nested2>
                    </root>
                '''
    test = xmlx.parser.from_string(xml)
    assert test.textlist(deep=True) == ['text1', 'text2', 'text3', 'text4']


def test_textdict_deep():
    xml = '''
                    <root a1="A1" a2="A2">
                    text
                        <nested>text1<deep>text2</deep></nested>
                        <nested2>text3<deep2>text4</deep2></nested2>
                    </root>
                '''
    test = xmlx.parser.from_string(xml)
    assert test.textdict(deep=True) == {'nested': 'text1', 'deep': 'text2', 'nested2': 'text3', 'deep2': 'text4'}


def test_textdict():
    xml = '''
                <container>
                    <tag1> xxx </tag1>
                    <tag2> yyy </tag2>
                    <tag3> zzz </tag3>
                </container>'''
    test = xmlx.parser.from_string(xml)
    assert test.textdict() == {'tag1': 'xxx', 'tag2': 'yyy', 'tag3': 'zzz'}


def test_textdict_nested():
    xml = '''
            <root a1="A1" a2="A2">
            text
                <nested><deep>text3</deep></nested>
            </root>
        '''
    test = xmlx.parser.from_string(xml)
    assert test.textdict('nested/deep') == {'deep': 'text3'}


def test_textdict_empty():
    xml = '''
            <root>
            </root>
        '''
    test = xmlx.parser.from_string(xml)
    assert test.textdict() == {}
