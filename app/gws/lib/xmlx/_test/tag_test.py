import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx


def test_simple():
    tag = xmlx.tag('name', 'text', {'a1': 'A1', 'a2': 'A2'})
    xml = tag.to_string()
    assert xml == '<name a1="A1" a2="A2">text</name>'


def test_nested():
    el = xmlx.tag(
        'root',
        'text',
        {'a1': 'A1', 'a2': 'A2'},
        [
            'nested',
            ['deep', 'text2'],
            'text3',
            ['single', {'b': 'B1'}],
        ],
    )

    xml = el.to_string()
    assert u.fxml(xml) == u.fxml("""
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
    el = xmlx.tag(
        'root',
        ['wms:foo'],
        ['{http://www.opengis.net/wfs/2.0}bar'],
    )

    xml = el.to_string()
    assert u.fxml(xml) == u.fxml('<root><wms:foo/><wfs:bar/></root>')

    xml = el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))

    assert u.fxml(xml) == u.fxml("""
        <root 
            xmlns:wfs="http://www.opengis.net/wfs/2.0" 
            xmlns:wms="http://www.opengis.net/wms">
            <wms:foo/>
            <wfs:bar/>
        </root>
    """)


def test_with_default_namespace():
    el = xmlx.tag(
        'root',
        ['wms:foo'],
        ['{http://www.opengis.net/wfs/2.0}bar'],
    )

    xml = el.to_string(
        gws.XmlOptions(
            withNamespaceDeclarations=True,
            defaultNamespace=xmlx.namespace.require('wms'),
        )
    )
    print(xml)

    assert u.fxml(xml) == u.fxml("""
        <root 
            xmlns="http://www.opengis.net/wms" 
            xmlns:wfs="http://www.opengis.net/wfs/2.0">
            <foo/>
            <wfs:bar/>
        </root>
    """)


def test_with_space():
    el = xmlx.tag('1 / 2 / 3')
    assert u.fxml(el.to_string()) == u.fxml("""
        <1>
            <2>
                <3/>
            </2>
        </1>
    """)


def test_text_str():
    el = xmlx.tag('root', 'text')
    assert u.fxml(el.to_string()) == u.fxml('<root>text</root>')


def test_text_int():
    el = xmlx.tag('root', 2)
    assert u.fxml(el.to_string()) == u.fxml('<root>2</root>')


def test_append_tuple2():
    el = xmlx.tag('root/nested', ('foo', 2))
    assert u.fxml(el.to_string()) == u.fxml("""
        <root>
            <nested>
                <foo>
                    2
                </foo>
            </nested>
        </root>
    """)


def test_append_tuple():
    el = xmlx.tag('root/nested', ('foo', 'bar'))
    assert u.fxml(el.to_string()) == u.fxml("""
        <root>
            <nested>
                <foo>
                    bar
                </foo>
            </nested>
        </root>
    """)


def test_child():
    child = xmlx.tag('child')
    el = xmlx.tag('root', child)
    assert u.fxml(el.to_string()) == u.fxml("""
        <root>
            <child/>
        </root>
    """)


def test_dict_attr():
    attr = {'foo': 1, 'bar': 2}
    el = xmlx.tag('root', attr)
    assert u.fxml(el.to_string()) == u.fxml('<root foo="1" bar="2"/>')


def test_list():
    list = ['foo', 'bar', 'foo2', 'bar2']
    el = xmlx.tag('root', list)
    assert u.fxml(el.to_string()) == u.fxml("""
        <root>
            <foo>
                barfoo2bar2
            </foo>
        </root>
    """)


def test_keywords():
    el = xmlx.tag('root', foo='bar')
    assert u.fxml(el.to_string()) == u.fxml('<root foo="bar"/>')


def test_tag():
    el = xmlx.tag(
        'geometry/gml:Point',
        {'gml:id': 'xy'},
        ['gml:coordinates', '12.345,56.789'],
        srsName=3857,
    )
    assert u.fxml(el.to_string()) == u.fxml("""
        <geometry>
            <gml:Point gml:id="xy" srsName="3857">
                <gml:coordinates>12.345,56.789</gml:coordinates>
            </gml:Point>
        </geometry>
    """)


def test_tag_uri():
    el = xmlx.tag(
        '{http://www.opengis.net/cat/csw/2.0.2}foo/nested',
        {'gml:id': 'xy'},
        ['gml:coordinates', '12.345,56.789'],
        srsName=3857,
    )
    assert u.fxml(el.to_string()) == u.fxml("""
        <csw:foo>
            <nested gml:id="xy" srsName="3857">
                <gml:coordinates>12.345,56.789</gml:coordinates>
            </nested>
        </csw:foo>
    """)
