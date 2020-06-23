import gws.config.loader
import gws.types as t

import _test.util as u

base = '_/cmd/assetHttpGetPath/projectUid/x/path/'

root = gws.config.loader.load()


def _render(src, content={}):
    tpl = root.create_object('gws.ext.template', t.Config(type='xml', text=src))
    return u.xml(tpl.render({}).content)


def test_simple():
    src = '''
        @tag abc
            @a numeric=123
            @a string1=abc
            @a string2="abc def"
            @t subtag1 subattr=123 subtext1
            @t subtag2 subattr=465 subtext2
        @end
    '''

    res = '''
        <abc numeric="123" string1="abc" string2="abc def">
            <subtag1 subattr="123">subtext1</subtag1>
            <subtag2 subattr="465">subtext2</subtag2>
        </abc>
    '''

    assert _render(src) == u.xml(res)

def test_custom_namespaces():
    src = '''
        @tag abc
            @xmlns foo http://foo.url default
            @xmlns bar http://bar.url
            @xmlns baz http://baz.url http://baz.schema
            @a simple=123
            @a bar:attr=123
            @t baz:subtag
        @end
    '''

    res = '''
        <abc
            xmlns:bar="http://bar.url"
            xmlns:baz="http://baz.url"
            xmlns="http://foo.url"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            simple="123"
            bar:attr="123"
            xsi:schemaLocation="http://baz.url http://baz.schema">
            <baz:subtag/>
        </abc>
    '''

    assert _render(src) == u.xml(res)


def test_std_namespaces():
    src = '''
        @tag abc
            @xmlns wms default
            @xmlns xlink
            @t subtag
        @end
    '''

    res = '''
        <abc
            xmlns="http://www.opengis.net/wms"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd http://www.w3.org/1999/xlink https://www.w3.org/XML/2008/06/xlink.xsd">
            <subtag/>
        </abc>
    '''

    assert _render(src) == u.xml(res)


def test_autodetect_namespaces():
    src = '''
        @tag abc
            @t wms:subtag gml:foo=bar xlink:href=123
        @end
    '''

    res = '''
        <abc
            xmlns:gml="http://www.opengis.net/gml/3.2"
            xmlns:wms="http://www.opengis.net/wms"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd http://www.w3.org/1999/xlink https://www.w3.org/XML/2008/06/xlink.xsd">
            <wms:subtag gml:foo="bar" xlink:href="123"/>
        </abc>
    '''

    assert _render(src) == u.xml(res)


def test_insert():
    src = '''
        @tag abc
            @t wms:subtag gml:foo=bar xlink:href=123
        @end
    '''

    res = '''
        <abc
            xmlns:gml="http://www.opengis.net/gml/3.2"
            xmlns:wms="http://www.opengis.net/wms"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd http://www.w3.org/1999/xlink https://www.w3.org/XML/2008/06/xlink.xsd">
            <wms:subtag gml:foo="bar" xlink:href="123"/>
        </abc>
    '''

    assert _render(src) == u.xml(res)
