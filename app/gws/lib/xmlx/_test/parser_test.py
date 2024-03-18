"""Tests for the parser module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx

xmlstr = '''
        <Root
                xmlns:gml="http://www.opengis.net/gml/3.0"
        >
            <a xmlns:foo="http://www.opengis.net/cat/csw">
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <foo:A TEST="gml2"/>
            </a>
        </Root>
        
    '''


def test_from_path():
    with open('/tmp/myfile.xml', 'w') as f:
        f.write(xmlstr)
        f.close()
        test = xmlx.parser.from_path('/tmp/myfile.xml')
        assert test.to_string(with_namespace_declarations=True) == '''<Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>'''


def test_from_path_case():
    with open('/tmp/myfile.xml', 'w') as f:
        f.write(xmlstr)
        f.close()
        test = xmlx.parser.from_path('/tmp/myfile.xml', case_insensitive=True)
        assert test.to_string() == '''<root>
            <a>
                <gml:a test="gml1"/>
                <a test="just"/>
                <a test="CASESENSITIVE"/>
                <csw:a test="gml2"/>
            </a>
        </root>'''


def test_from_path_compact():
    with open('/tmp/myfile.xml', 'w') as f:
        f.write(xmlstr)
        f.close()
        test = xmlx.parser.from_path('/tmp/myfile.xml', compact_whitespace=True)
        assert test.to_string() == '<Root><a><gml:a test="gml1"/><a Test="just"/><A TEST="CASESENSITIVE"/><csw:A TEST="gml2"/></a></Root>'


# no clue what it does
def test_from_path_normalize():
    with open('/tmp/myfile.xml', 'w') as f:
        f.write(xmlstr)
        f.close()
        test = xmlx.parser.from_path('/tmp/myfile.xml', normalize_namespaces=True)
        assert test.to_string(with_namespace_declarations=True) == '''<Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>'''


def test_from_path_remove():
    with open('/tmp/myfile.xml', 'w') as f:
        f.write(xmlstr)
        f.close()
        test = xmlx.parser.from_path('/tmp/myfile.xml', remove_namespaces=True)
        assert test.to_string(with_namespace_declarations=True) == '''<Root>
            <a>
                <a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <A TEST="gml2"/>
            </a>
        </Root>'''


def test_from_string():
    test = xmlx.parser.from_string(xmlstr)
    assert test.to_string(with_namespace_declarations=True) == '''<Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>'''


def test_from_string_case():
    test = xmlx.parser.from_string(xmlstr, case_insensitive=True)
    assert test.to_string() == '''<root>
            <a>
                <gml:a test="gml1"/>
                <a test="just"/>
                <a test="CASESENSITIVE"/>
                <csw:a test="gml2"/>
            </a>
        </root>'''


def test_from_string_compact():
    test = xmlx.parser.from_string(xmlstr, compact_whitespace=True)
    assert test.to_string() == '<Root><a><gml:a test="gml1"/><a Test="just"/><A TEST="CASESENSITIVE"/><csw:A TEST="gml2"/></a></Root>'


def test_from_string_remove():
    test = xmlx.parser.from_string(xmlstr, remove_namespaces=True)
    assert test.to_string() == '''<Root>
            <a>
                <a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <A TEST="gml2"/>
            </a>
        </Root>'''



def test_from_string_normalize():
    test = xmlx.parser.from_string(xmlstr, normalize_namespaces=True)
    assert test.to_string(with_namespace_declarations=True) == '''<Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>'''


# encoded in iso-8859-1 and declared utf-8 but declaration still shows utf-8
def test_from_string_decode_1():
    xml_decode = '<?xml version="1.0" encoding="UTF-8"?>\n' + xmlstr
    test = xmlx.parser.from_string(xml_decode)
    assert test.to_string(with_xml_declaration=True, with_namespace_declarations=True) == ('''<?xml version="1.0" encoding="UTF-8"?><Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>''')


# encoded in utf-8 and declared as iso-8859-1 encoding but shows utf-8 encoding
def test_from_string_decode_2():
    xml_decode = '<?xml version="1.0" encoding="iso-8859-1"?>\n' + xmlstr
    test = xmlx.parser.from_string(xml_decode)
    assert test.to_string(with_xml_declaration=True, with_namespace_declarations=True) == ('''<?xml version="1.0" encoding="UTF-8"?><Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>''')


def test_from_bytes():
    test = xmlx.parser.from_string(xmlstr.encode('utf-8'))
    assert test.to_string(with_namespace_declarations=True) == '''<Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>'''


def test_from_bytes_case():
    test = xmlx.parser.from_string(xmlstr.encode('utf-8'), case_insensitive=True)
    assert test.to_string() == '''<root>
            <a>
                <gml:a test="gml1"/>
                <a test="just"/>
                <a test="CASESENSITIVE"/>
                <csw:a test="gml2"/>
            </a>
        </root>'''


def test_from_bytes_compact():
    test = xmlx.parser.from_string(xmlstr.encode('utf-8'), compact_whitespace=True)
    assert test.to_string() == '<Root><a><gml:a test="gml1"/><a Test="just"/><A TEST="CASESENSITIVE"/><csw:A TEST="gml2"/></a></Root>'


def test_from_bytes_remove():
    test = xmlx.parser.from_string(xmlstr.encode('utf-8'), remove_namespaces=True)
    assert test.to_string() == '''<Root>
            <a>
                <a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <A TEST="gml2"/>
            </a>
        </Root>'''



def test_from_bytes_normalize():
    test = xmlx.parser.from_string(xmlstr.encode('utf-8'), normalize_namespaces=True)
    assert test.to_string(with_namespace_declarations=True) == '''<Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>'''


# encoded in iso-8859-1 and declared utf-8 but declaration still shows utf-8
def test_from_bytes_decode_1():
    xml_decode = '<?xml version="1.0" encoding="UTF-8"?>\n' + xmlstr
    test = xmlx.parser.from_string(xml_decode.encode('iso-8859-1'))
    assert test.to_string(with_xml_declaration=True, with_namespace_declarations=True) == ('''<?xml version="1.0" encoding="UTF-8"?><Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>''')


# encoded in utf-8 and declared as iso-8859-1 encoding but shows utf-8 encoding
def test_from_bytes_decode_2():
    xml_decode = '<?xml version="1.0" encoding="iso-8859-1"?>\n' + xmlstr
    test = xmlx.parser.from_string(xml_decode.encode('UTF-8'))
    assert test.to_string(with_xml_declaration=True, with_namespace_declarations=True) == ('''<?xml version="1.0" encoding="UTF-8"?><Root xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:gml="http://www.opengis.net/gml/3.2">
            <a>
                <gml:a test="gml1"/>
                <a Test="just"/>
                <A TEST="CASESENSITIVE"/>
                <csw:A TEST="gml2"/>
            </a>
        </Root>''')


def test_parser_convert_name():
    p = xmlx.parser._ParserTarget(case_insensitive=False,
                                  compact_whitespace=False,
                                  normalize_namespaces=False,
                                  remove_namespaces=False)
    assert p.convert_name('foo') == 'foo'
    p.remove_namespaces = True
    assert p.convert_name('{http://www.opengis.net/cat/csw}foo') == 'foo'
    assert p.convert_name('gml:foo') == 'foo'
    p.remove_namespaces = False
    p.normalize_namespaces = True
    assert p.convert_name('{http://www.opengis.net/gml/999}foo') == '{http://www.opengis.net/gml}foo'
    assert p.convert_name('gml:foo') == '{}foo'
    p.normalize_namespaces = False
    assert p.convert_name('{http://www.opengis.net/cat/csw}foo') == '{http://www.opengis.net/cat/csw}foo'
    assert p.convert_name('gml:foo') == '{}foo'


def test_parser_make():
    p = xmlx.parser._ParserTarget(case_insensitive=False,
                                  compact_whitespace=False,
                                  normalize_namespaces=False,
                                  remove_namespaces=False)
    t = 'tag'
    a = {'attribute': 'foo'}
    assert p.make(t, a).to_string() == '<tag attribute="foo"/>'


def test_parser_make_noattr():
    p = xmlx.parser._ParserTarget(case_insensitive=False,
                                  compact_whitespace=False,
                                  normalize_namespaces=False,
                                  remove_namespaces=False)
    t = 'tag'
    a = {}
    assert p.make(t, a).to_string() == '<tag/>'


def test_parser_flush():
    p = xmlx.parser._ParserTarget(case_insensitive=False,
                                  compact_whitespace=False,
                                  normalize_namespaces=False,
                                  remove_namespaces=False)
    p.buf = ['a']
    p.stack = [xmlx.parser.from_string('<Root><a test="attr"/></Root>')]
    p.flush()
    assert p.buf == []
    assert p.stack[-1].text == ''


def test_parser_start():
    p = xmlx.parser._ParserTarget(case_insensitive=False,
                                  compact_whitespace=False,
                                  normalize_namespaces=False,
                                  remove_namespaces=False)
    p.start('root', {})
    p.start('child', {'test': 'a'})
    assert p.stack[0].to_string() == '<root><child test="a"/></root>'
    assert p.stack[1].to_string() == '<child test="a"/>'


def test_parser_end():
    p = xmlx.parser._ParserTarget(case_insensitive=False,
                                  compact_whitespace=False,
                                  normalize_namespaces=False,
                                  remove_namespaces=False)
    p.start('root', {})
    p.start('child', {'test':'a'})
    p.data('foo')
    p.data(' bar')
    p.end('child')
    assert p.buf == []
    assert len(p.stack) == 1
    assert p.stack[0].to_string() == '<root><child test="a">foo bar</child></root>'


def test_parser_data():
    p = xmlx.parser._ParserTarget(case_insensitive=False,
                                  compact_whitespace=False,
                                  normalize_namespaces=False,
                                  remove_namespaces=False)
    p.data('d1')
    p.data('d2')
    p.data('d3')
    assert p.buf == ['d1', 'd2', 'd3']


def test_parser_close():
    p = xmlx.parser._ParserTarget(case_insensitive=False,
                                  compact_whitespace=False,
                                  normalize_namespaces=False,
                                  remove_namespaces=False)
    p.root = 'test'
    assert p.close() == 'test'
