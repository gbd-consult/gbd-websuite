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


def test_from_string_case():
    s = '<foo><BAR a1="x" A2="y"/></foo>'
    doc = xmlx.parser.from_string(s, gws.XmlOptions(caseInsensitive=True))
    assert _noempty(doc.to_dict()) == {'children': [{'attrib': {'a1': 'x', 'a2': 'y'}, 'tag': 'bar'}], 'tag': 'foo'}


def test_from_string_compact():
    s = '<foo> x <bar>   abc   </bar> y </foo>'
    doc = xmlx.parser.from_string(s, gws.XmlOptions(compactWhitespace=True))
    assert _noempty(doc.to_dict()) == {'children': [{'tag': 'bar', 'text': 'abc', 'tail': 'y'}], 'tag': 'foo', 'text': 'x'}


def test_from_string_namespaces():
    s = '<ns:a xmlns:ns="http://ns1"><b xmlns="http://ns2" ns:x="1"><c/></b></ns:a>'
    doc = xmlx.parser.from_string(s, gws.XmlOptions(removeNamespaces=False))
    assert _noempty(doc.to_dict()) == {'tag': '{http://ns1}a', 'children': [{'tag': '{http://ns2}b', 'attrib': {'{http://ns1}x': '1'}, 'children': [{'tag': '{http://ns2}c'}]}]}
    doc = xmlx.parser.from_string(s, gws.XmlOptions(removeNamespaces=True))
    assert _noempty(doc.to_dict()) == {'tag': 'a', 'children': [{'tag': 'b', 'attrib': {'x': '1'}, 'children': [{'tag': 'c'}]}]}


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
    s = b'<?xml version="1.0" encoding="UTF-8"?><foo>\xe4\xdf</foo>'
    doc = xmlx.parser.from_string(s)
    assert _noempty(doc.to_dict()) == {'tag': 'foo', 'text': 'äß'}


def _noempty(d):
    if isinstance(d, dict):
        return {k: _noempty(v) for k, v in d.items() if v not in (None, '', [], {})}
    if isinstance(d, list):
        return [_noempty(v) for v in d if v not in (None, '', [], {})]
    return d
