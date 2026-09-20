"""Tests for the parser module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx


def test_from_path(tmpdir):
    s = '<foo><bar/></foo>'
    tf = tmpdir.join('test.xml')
    tf.write(s)
    doc = xmlx.parser.from_path(str(tf))
    assert _noempty(doc.to_dict()) == {'children': [{'tag': 'bar'}], 'tag': 'foo'}


def test_from_string():
    s = '<foo><bar/></foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'children': [{'tag': 'bar'}], 'tag': 'foo'}


def test_text_and_tail():
    s = '<foo>a<bar>b</bar>c<baz/>d</foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {
        'tag': 'foo',
        'text': 'a',
        'children': [{'tag': 'bar', 'text': 'b', 'tail': 'c'}, {'tag': 'baz', 'tail': 'd'}],
    }


def test_whitespace_is_kept():
    s = '<foo> x <bar>   abc   </bar> y </foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'children': [{'tag': 'bar', 'text': '   abc   ', 'tail': ' y '}], 'tag': 'foo', 'text': ' x '}


def test_from_string_compact():
    s = '<foo> x <bar>   abc   </bar> y </foo>'
    doc = xmlx.parser.from_string(s, gws.XmlOptions(compactWhitespace=True))
    assert _noempty(doc.to_dict()) == {'children': [{'tag': 'bar', 'text': 'abc', 'tail': 'y'}], 'tag': 'foo', 'text': 'x'}


def test_namespaces_are_stripped():
    s = '<ns:a xmlns:ns="http://ns1"><b xmlns="http://ns2" ns:x="1"><ns:c/></b></ns:a>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'a', 'children': [{'tag': 'b', 'attrib': {'x': '1'}, 'children': [{'tag': 'c'}]}]}
    assert doc.name == 'a'


def test_namespace_declarations_are_kept():
    s = '<ns:a xmlns:ns="http://ns1"><b xmlns="http://ns2" xmlns:other="http://ns3"><c/></b></ns:a>'
    doc = xmlx.parser.from_string(s)
    assert {k: v.uri for k, v in doc.namespaces.items()} == {'ns': 'http://ns1'}
    assert {k: v.uri for k, v in doc[0].namespaces.items()} == {'': 'http://ns2', 'other': 'http://ns3'}
    assert doc[0][0].namespaces == {}
    assert doc.namespaces['ns'].xmlns == 'ns'


def test_undeclared_prefix_is_accepted():
    s = '<a><gml:b gml:x="1"/></a>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'a', 'children': [{'tag': 'b', 'attrib': {'x': '1'}}]}


def test_doctype_is_accepted():
    s = '<!DOCTYPE a SYSTEM "http://example.com/a.dtd" [<!ELEMENT a EMPTY>]><a/>'
    doc = xmlx.parser.from_string(s)
    assert doc.tag == 'a'


def test_entity_declaration_is_rejected():
    s = '<!DOCTYPE a [<!ENTITY x "hi">]><a>&x;</a>'
    with u.raises(xmlx.ParseError):
        xmlx.parser.from_string(s)


def test_predefined_entities():
    s = '<a x="&lt;&amp;&quot;">&lt;&gt;&amp;&#228;&#xE4;</a>'
    doc = xmlx.parser.from_string(s)
    assert doc.text == '<>&ää'
    assert doc.get('x') == '<&"'


def test_comments_and_pis_are_dropped():
    s = '<?xml version="1.0"?><?pi x?><!-- c --><a><!-- c -->x<?pi y?><b/></a>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'a', 'text': 'x', 'children': [{'tag': 'b'}]}


def test_cdata():
    s = '<a><![CDATA[<x> & y]]></a>'
    doc = xmlx.parser.from_string(s)
    assert doc.text == '<x> & y'


def test_not_well_formed():
    with u.raises(xmlx.ParseError):
        xmlx.parser.from_string('<a><b></a>')
    with u.raises(xmlx.ParseError):
        xmlx.parser.from_string('<a>x & y</a>')
    with u.raises(xmlx.ParseError):
        xmlx.parser.from_string('')


def test_decode_str_valid_encoding():
    s = '<?xml version="1.0" encoding="UTF-8"?><foo>äß</foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'foo', 'text': 'äß'}


def test_decode_str_invalid_encoding():
    s = '<?xml version="1.0" encoding="iso-8859-1"?><foo>äß</foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'foo', 'text': 'äß'}


def test_decode_bytes_valid_encoding():
    s = b'<?xml version="1.0" encoding="UTF-8"?><foo>\xc3\xa4\xc3\x9f</foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'foo', 'text': 'äß'}
    s = b'<?xml version="1.0" encoding="iso-8859-1"?><foo>\xe4\xdf</foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'foo', 'text': 'äß'}


def test_decode_bytes_invalid_encoding():
    # declared UTF-8, actually Latin-1
    s = b'<?xml version="1.0" encoding="UTF-8"?><foo>\xe4\xdf</foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'foo', 'text': 'äß'}
    # declared Latin-1, actually UTF-8
    s = b'<?xml version="1.0" encoding="iso-8859-1"?><foo>\xc3\xa4\xc3\x9f</foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'foo', 'text': 'äß'}


def test_decode_bytes_other_encoding():
    s = '<?xml version="1.0" encoding="cp1251"?><foo>привет</foo>'.encode('cp1251')
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'foo', 'text': 'привет'}


def test_decode_bytes_no_declaration():
    doc = xmlx.parser.from_string(b'<foo>\xc3\xa4</foo>')
    assert doc.text == 'ä'
    doc = xmlx.parser.from_string(b'<foo>\xe4</foo>')
    assert doc.text == 'ä'


def test_bom():
    doc = xmlx.parser.from_string(b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?><foo>\xc3\xa4</foo>')
    assert doc.text == 'ä'
    doc = xmlx.parser.from_string('﻿<?xml version="1.0"?><foo>ä</foo>')
    assert doc.text == 'ä'


def _noempty(d):
    if isinstance(d, dict):
        return {k: _noempty(v) for k, v in d.items() if v not in (None, '', [], {})}
    if isinstance(d, list):
        return [_noempty(v) for v in d if v not in (None, '', [], {})]
    return d
