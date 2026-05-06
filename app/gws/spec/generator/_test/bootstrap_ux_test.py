"""Tests for the bootstrap_ux CLI."""

import json
import os
import textwrap

import pytest

from gws.spec.generator import bootstrap_ux as B


@pytest.mark.parametrize(
    'ident,expected',
    [
        ('bindDN', 'Bind-DN'),
        # Password term is not in the German domain lookup — heuristic outputs the
        # English form; humans translate it during the polish step.
        ('bindPassword', 'Bind-Password'),
        ('schemaCacheLifeTime', 'Schema-Cache-Life-Time'),
        ('useCanvasExtent', 'Canvas-Extent verwenden'),
        ('host', 'Host'),
        ('port', 'Port'),
        ('wmsUrl', 'WMS-URL'),
        ('enableCors', 'CORS aktivieren'),
        ('tcLifeTime', 'TC-Life-Time'),
        ('httpsOnly', 'HTTPS-Only'),
    ],
)
def test_label_from_ident(ident, expected):
    assert B._label_from_ident(ident, 'de') == expected


def test_split_camel_basic():
    assert B._split_camel('bindDN') == ['bind', 'DN']
    assert B._split_camel('schemaCacheLifeTime') == ['schema', 'Cache', 'Life', 'Time']


def test_guess_complexity():
    assert B._guess_complexity('host') == 'basic'
    assert B._guess_complexity('port') == 'basic'
    assert B._guess_complexity('cacheTime') == 'advanced'
    assert B._guess_complexity('connectionPool') == 'advanced'
    assert B._guess_complexity('mySetting') == ''


def test_first_sentence():
    assert B._first_sentence('Hello world. Second.') == 'Hello world.'
    assert B._first_sentence('No punctuation') == 'No punctuation'
    assert B._first_sentence('') == ''
    assert B._first_sentence('Multiline\n\nis collapsed. Then more.') == 'Multiline is collapsed.'


def test_strip_lang_prefix():
    assert B._strip_lang_prefix('[en]hello') == 'hello'
    assert B._strip_lang_prefix('[de] Hallo') == 'Hallo'
    assert B._strip_lang_prefix('plain') == 'plain'


def test_module_prefix_from_dir(tmp_path):
    p = tmp_path / 'app' / 'gws' / 'plugin' / 'foo'
    p.mkdir(parents=True)
    assert B._module_prefix_from_dir(str(p)) == 'gws.plugin.foo'

    p2 = tmp_path / 'app' / 'gws' / 'base' / 'auth' / 'manager'
    p2.mkdir(parents=True)
    assert B._module_prefix_from_dir(str(p2)) == 'gws.base.auth.manager'


def test_module_prefix_from_dir_rejects_outside():
    with pytest.raises(ValueError):
        B._module_prefix_from_dir('/tmp/not-a-gws-tree')


def _fake_specs(types):
    return {'serverTypes': types}


def test_collect_classes_filters_by_prefix():
    specs = _fake_specs([
        {'c': 'CLASS', 'name': 'gws.plugin.foo.Config', 'modName': 'gws.plugin.foo'},
        {'c': 'CLASS', 'name': 'gws.plugin.foo.bar.Config', 'modName': 'gws.plugin.foo.bar'},
        {'c': 'CLASS', 'name': 'gws.plugin.other.Config', 'modName': 'gws.plugin.other'},
        {'c': 'PROPERTY', 'name': 'gws.plugin.foo.Config.host', 'modName': 'gws.plugin.foo'},
    ])
    result = B.collect_classes(specs, 'gws.plugin.foo')
    names = [t['name'] for t in result]
    assert names == ['gws.plugin.foo.Config', 'gws.plugin.foo.bar.Config']


def test_collect_properties_for_class():
    specs = _fake_specs([
        {
            'c': 'PROPERTY',
            'name': 'gws.plugin.foo.Config.host',
            'tOwner': 'gws.plugin.foo.Config',
            'modName': 'gws.plugin.foo',
            'doc': 'Host name.',
        },
        {
            'c': 'PROPERTY',
            # Inherited — defined on a parent class
            'name': 'gws.base.x.Config.access',
            'tOwner': 'gws.plugin.foo.Config',
            'modName': 'gws.base.x',
        },
        {
            'c': 'PROPERTY',
            'name': 'gws.plugin.foo.Config.port',
            'tOwner': 'gws.plugin.foo.Config',
            'modName': 'gws.plugin.foo',
        },
    ])
    props = B.collect_properties_for(specs, 'gws.plugin.foo.Config', 'gws.plugin.foo')
    names = [p['name'] for p in props]
    assert names == ['gws.plugin.foo.Config.host', 'gws.plugin.foo.Config.port']


def test_render_skeleton_includes_classes_and_props():
    specs = _fake_specs([
        {
            'c': 'CLASS',
            'name': 'gws.plugin.foo.Config',
            'modName': 'gws.plugin.foo',
            'doc': 'Foo provider config. More text.',
        },
        {
            'c': 'PROPERTY',
            'name': 'gws.plugin.foo.Config.host',
            'tOwner': 'gws.plugin.foo.Config',
            'modName': 'gws.plugin.foo',
            'doc': 'Database host.',
        },
        {
            'c': 'PROPERTY',
            'name': 'gws.plugin.foo.Config.cachePool',
            'tOwner': 'gws.plugin.foo.Config',
            'modName': 'gws.plugin.foo',
            'doc': '',
        },
    ])
    out = B.render_skeleton(specs, 'gws.plugin.foo', ['de', 'en'])

    assert '[de]' in out
    assert '[en]' in out
    assert 'gws.plugin.foo.Config.label = Foo' in out  # label from module name "foo"
    assert 'gws.plugin.foo.Config.purpose = Foo provider config.' in out
    assert 'gws.plugin.foo.Config.host.label = Host' in out
    assert 'gws.plugin.foo.Config.host.purpose = Database host.' in out
    assert 'gws.plugin.foo.Config.host.complexity = basic' in out
    assert 'gws.plugin.foo.Config.cachePool.complexity = advanced' in out


def test_render_skeleton_empty_when_no_classes():
    specs = _fake_specs([])
    out = B.render_skeleton(specs, 'gws.plugin.foo', ['de'])
    assert 'No CLASS types found' in out


def test_bootstrap_writes_file_only_if_missing(tmp_path):
    # Build a fake plugin tree under .../app/gws/plugin/foo
    plugin_dir = tmp_path / 'app' / 'gws' / 'plugin' / 'foo'
    plugin_dir.mkdir(parents=True)

    specs_path = tmp_path / 'specs.json'
    specs_path.write_text(json.dumps(_fake_specs([
        {'c': 'CLASS', 'name': 'gws.plugin.foo.Config', 'modName': 'gws.plugin.foo', 'doc': 'X.'},
    ])))

    text, written = B.bootstrap_plugin(str(plugin_dir), ['de'], apply=True, specs_path=str(specs_path))
    assert written is not None
    assert os.path.exists(written)
    on_disk = open(written).read()
    assert 'gws.plugin.foo.Config.label' in on_disk

    # Second call must NOT overwrite
    _, written2 = B.bootstrap_plugin(str(plugin_dir), ['de'], apply=True, specs_path=str(specs_path))
    assert written2 is None  # refused
    assert open(written).read() == on_disk  # unchanged


def test_bootstrap_dry_run_does_not_write(tmp_path):
    plugin_dir = tmp_path / 'app' / 'gws' / 'plugin' / 'foo'
    plugin_dir.mkdir(parents=True)

    specs_path = tmp_path / 'specs.json'
    specs_path.write_text(json.dumps(_fake_specs([
        {'c': 'CLASS', 'name': 'gws.plugin.foo.Config', 'modName': 'gws.plugin.foo', 'doc': ''},
    ])))

    text, written = B.bootstrap_plugin(str(plugin_dir), ['de'], apply=False, specs_path=str(specs_path))
    assert written is None
    assert 'gws.plugin.foo.Config.label' in text
    assert not os.path.exists(plugin_dir / '_doc' / 'ux.ini')
