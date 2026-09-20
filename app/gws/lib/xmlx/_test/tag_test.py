import datetime

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx
from gws.lib.xmlx import tag


def test_simple():
    el = tag('name', 'text', {'a1': 'A1', 'a2': 'A2'})
    xml = el.to_string()
    assert xml == '<name a1="A1" a2="A2">text</name>'


def test_nested():
    el = tag(
        'root',
        'text',
        {'a1': 'A1', 'a2': 'A2'},
        tag(
            'nested',
            tag('deep', 'text2'),
            'text3',
            tag('single', {'b': 'B1'}),
        ),
    )

    xml = el.to_string()
    u.check.xml(xml, """
        <root a1="A1" a2="A2">
        text
            <nested>
                <deep>text2</deep>
                text3
                <single b="B1"/>
            </nested>
        </root>
    """)


def test_with_namespaces():
    el = tag(
        'root',
        tag('WMS:foo'),
        tag('WFS:bar'),
    )
    assert el[0].tag == '{http://www.opengis.net/wms}foo'
    assert el[0].name == 'foo'

    xml = el.to_string()
    u.check.xml(xml, '<root><wms:foo/><wfs:bar/></root>')

    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))

    u.check.xml(xml, """
        <root
            xmlns:wfs="http://www.opengis.net/wfs/2.0"
            xmlns:wms="http://www.opengis.net/wms">
            <wms:foo/>
            <wfs:bar/>
        </root>
    """)


def test_with_default_namespace():
    el = tag(
        'root',
        tag('WMS:foo'),
        tag('WFS:bar'),
    )

    xml = el.to_string(
        gws.XmlOptions(
            defaultNamespace=xmlx.namespace.ns.WFS,
            withNamespaceDeclarations=True,
        )
    )

    u.check.xml(xml, """
        <root
            xmlns="http://www.opengis.net/wfs/2.0"
            xmlns:wms="http://www.opengis.net/wms">
            <wms:foo/>
            <bar/>
        </root>
    """)


def test_with_space():
    el = tag('a / b / c')
    u.check.xml(el.to_string(), """
        <a>
            <b>
                <c/>
            </b>
        </a>
    """)


def test_text_str():
    el = tag('root', 'text')
    u.check.xml(el.to_string(), '<root>text</root>')


def test_text_int():
    el = tag('root', 2)
    u.check.xml(el.to_string(), '<root>2</root>')


def test_text_bool_and_date():
    el = tag('root', True, ' ', datetime.date(2020, 1, 2))
    u.check.xml(el.to_string(), '<root>true 2020-01-02</root>')


def test_text_and_tail():
    el = tag('root', 'a', tag('x'), 'b', tag('y'), 'c')
    assert el.text == 'a'
    assert el[0].tail == 'b'
    assert el[1].tail == 'c'
    u.check.xml(el.to_string(), '<root>a<x/>b<y/>c</root>')


def test_child():
    child = tag('child')
    el = tag('root', child)
    assert el[0] is child
    u.check.xml(el.to_string(), '<root><child/></root>')


def test_dict_attr():
    el = tag('root', {'a': 'b', 'c': None, 'd': 1})
    u.check.xml(el.to_string(), '<root a="b" d="1"/>')


def test_iterables_are_spread():
    el = tag('root', ['foo', 'bar'], (tag('x'), 'baz'), (tag(f'y{i}') for i in range(2)))
    u.check.xml(el.to_string(), '<root>foobar<x/>baz<y0/><y1/></root>')


def test_nested_iterables():
    el = tag('root', [[tag('a'), [tag('b'), ['t']]]])
    u.check.xml(el.to_string(), '<root><a/><b/>t</root>')


def test_none_is_ignored():
    el = tag('root', None, [None, tag('a'), None])
    u.check.xml(el.to_string(), '<root><a/></root>')


def test_keywords():
    el = tag('root', foo='bar')
    u.check.xml(el.to_string(), '<root foo="bar"/>')


def test_invalid_argument():
    with u.raises(xmlx.BuildError):
        tag('root', gws.Data(x=1))
    with u.raises(xmlx.BuildError):
        tag('root', object())


def test_invalid_name():
    with u.raises(xmlx.BuildError):
        tag('')
    with u.raises(xmlx.BuildError):
        tag('a//b')


def test_unknown_namespace_id():
    with u.raises(xmlx.NamespaceError):
        tag('gml:Point')
    with u.raises(xmlx.NamespaceError):
        tag('a', {'gml:id': 1})


def test_clark_names():
    el = tag('{http://x}a', {'{http://y}b': 1})
    assert el.tag == '{http://x}a'
    assert el.name == 'a'
    assert el.attrib == {'{http://y}b': 1}


def test_tag():
    el = tag(
        'geometry/GML:Point',
        {'GML:id': 'xy'},
        tag('GML:coordinates', '12.345,56.789'),
        srsName=3857,
    )
    u.check.xml(el.to_string(), """
        <geometry>
            <gml:Point gml:id="xy" srsName="3857">
                <gml:coordinates>12.345,56.789</gml:coordinates>
            </gml:Point>
        </geometry>
    """)
