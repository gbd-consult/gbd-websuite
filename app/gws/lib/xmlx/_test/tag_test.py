import gws
import gws.types as t
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
        ]
    )

    xml = el.to_string()
    assert xml == u.fxml('''
        <root a1="A1" a2="A2">
        text
            <nested>
                <deep>text2</deep>
                text3
                <single b="B1"/>
            </nested>
        </root>
    ''')


def test_with_namespaces():
    el = xmlx.tag(
        'root',
        ['wms:foo'],
        ['{http://www.opengis.net/wfs}bar'],
    )

    xml = el.to_string()
    assert xml == u.fxml('<root><wms:foo/><wfs:bar/></root>')

    xml = el.to_string(with_namespace_declarations=True)
    assert xml == u.fxml('''
        <root 
            xmlns:wfs="http://www.opengis.net/wfs/2.0" 
            xmlns:wms="http://www.opengis.net/wms"
        >
            <wms:foo/>
            <wfs:bar/>
        </root>
    ''')


def test_with_default_namespace():
    el = xmlx.tag(
        'root',
        {'xmlns': 'wms'},
        ['wms:foo'],
        ['{http://www.opengis.net/wfs}bar'],
    )

    xml = el.to_string(with_namespace_declarations=True)
    assert xml == u.fxml('''
        <root 
            xmlns="http://www.opengis.net/wms" 
            xmlns:wfs="http://www.opengis.net/wfs/2.0"
        >
            <foo/>
            <wfs:bar/>
        </root>
    ''')
