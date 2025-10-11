"""Tests for the serializer module"""

import gws
import gws.test.util as u

import gws.lib.xmlx as xmlx


def test_to_list():
    el = xmlx.tag(
        'a/b/c',
        {'attr1': 'a'},
        ['sub', 'subtext'],
        attr3='3',
    )
    assert el.to_list() == ['a', ['b', ['c', {'attr1': 'a', 'attr3': '3'}, ['sub', 'subtext']]]]


def test_to_string_with_namespaces():
    aaa_ns = xmlx.namespace.from_args(uid='aaa', xmlns='aaa', uri='http://aaa')
    bbb_ns = xmlx.namespace.from_args(uid='bbb', xmlns='bbb', uri='http://bbb')

    el = xmlx.tag(
        'aaa:a/bbb:b',
        {
            'a1': 'A1',
            'aaa:a2': 'A2',
        },
        ['{http://bbb}sub'],
    )

    opts = gws.XmlOptions(
        namespaces={
            'aaa': aaa_ns,
            'bbb': bbb_ns,
        },
        withNamespaceDeclarations=True,
    )
    xml = el.to_string(opts)
    assert u.fxml(xml) == u.fxml("""
        <aaa:a xmlns:aaa="http://aaa" xmlns:bbb="http://bbb">
            <bbb:b a1="A1" aaa:a2="A2">
                <bbb:sub/>
            </bbb:b>
        </aaa:a>
    """)

def test_to_string_with_unknown_namespace():
    el = xmlx.tag(
        'aaa:a/bbb:b',
    )

    with u.raises(Exception):
        el.to_string()


def test_to_string_with_default_namespace():
    aaa_ns = xmlx.namespace.from_args(uid='aaa', uri='http://aaa')
    bbb_ns = xmlx.namespace.from_args(uid='bbb', uri='http://bbb')

    el = xmlx.tag(
        'aaa:a/bbb:b',
        {
            'a1': 'A1',
            'aaa:a2': 'A2',
        },
        ['bbb:sub'],
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
    assert u.fxml(xml) == u.fxml("""
        <a xmlns="http://aaa" xmlns:bbb="http://bbb">
            <bbb:b a1="A1" a2="A2">
                <bbb:sub/>
            </bbb:b>
        </a>
    """)

