"""Tests for the ux-strings collection pipeline.

These tests run without the docker stack — they build a minimal Generator
fixture in a tmp_path tree and exercise the strings module directly.
"""

import textwrap

import pytest

from gws.spec.generator import base, strings
from gws.spec.core import c, make_type


def _gen(tmp_path):
    g = base.Generator()
    g.rootDir = str(tmp_path)
    g.outDir = str(tmp_path / 'out')
    return g


def _write(tmp_path, rel, body):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).strip() + '\n', encoding='utf-8')
    return p


def test_collect_ux_parses_ini(tmp_path):
    _write(
        tmp_path,
        'plugin/foo/_doc/ux.ini',
        """
        [de]
        gws.plugin.foo.Config.label = Foo-Provider
        gws.plugin.foo.Config.purpose = Verbindet das Foo-System mit der WebSuite.
        gws.plugin.foo.Config.complexity = intermediate
        gws.plugin.foo.Config.host.label = Hostname

        [en]
        gws.plugin.foo.Config.label = Foo provider
        gws.plugin.foo.Config.purpose = Connects the Foo system to the WebSuite.
        """,
    )

    gen = _gen(tmp_path)
    ux = strings.collect_ux(gen)

    assert ux['de']['gws.plugin.foo.Config']['label'] == 'Foo-Provider'
    assert ux['de']['gws.plugin.foo.Config']['complexity'] == 'intermediate'
    assert ux['de']['gws.plugin.foo.Config.host']['label'] == 'Hostname'
    assert ux['en']['gws.plugin.foo.Config']['purpose'].startswith('Connects')


def test_collect_ux_no_files_returns_empty(tmp_path):
    gen = _gen(tmp_path)
    ux = strings.collect_ux(gen)
    assert ux == {}


def test_ux_ini_overrides_docstring_marker(tmp_path):
    """Marker collection runs first; ux.ini merges on top with priority."""
    gen = _gen(tmp_path)

    # Synthesise a parsed type with a docstring marker for complexity.
    typ = make_type(
        {
            'c': c.CLASS,
            'uid': 'gws.plugin.foo.Config',
            'name': 'gws.plugin.foo.Config',
            'doc': 'Foo config.\n\n:complexity: advanced\n:seeAlso: gws.plugin.bar.Config\n',
        }
    )
    gen.typeDict[typ.uid] = typ

    # Markers extracted from the docstring populate gen.uxStrings first.
    gen.uxStrings = strings.collect_docstring_markers(gen)
    assert gen.uxStrings['en']['gws.plugin.foo.Config']['complexity'] == 'advanced'
    assert gen.uxStrings['en']['gws.plugin.foo.Config']['seeAlso'] == 'gws.plugin.bar.Config'

    # Now an ux.ini overrides complexity but leaves seeAlso untouched.
    _write(
        tmp_path,
        'plugin/foo/_doc/ux.ini',
        """
        [en]
        gws.plugin.foo.Config.complexity = basic
        gws.plugin.foo.Config.purpose = Adapter for the Foo system.
        """,
    )

    gen.uxStrings = strings.collect_ux(gen)

    entry = gen.uxStrings['en']['gws.plugin.foo.Config']
    assert entry['complexity'] == 'basic'  # ux.ini wins
    assert entry['seeAlso'] == 'gws.plugin.bar.Config'  # marker survives
    assert entry['purpose'] == 'Adapter for the Foo system.'


def test_unknown_ux_field_is_skipped(tmp_path, caplog):
    _write(
        tmp_path,
        'plugin/foo/_doc/ux.ini',
        """
        [en]
        gws.plugin.foo.Config.label = Foo
        gws.plugin.foo.Config.notARealField = bogus
        """,
    )

    gen = _gen(tmp_path)
    ux = strings.collect_ux(gen)

    entry = ux['en']['gws.plugin.foo.Config']
    assert entry == {'label': 'Foo'}
    assert 'notARealField' not in entry


def test_apply_ux_to_variants_sets_doc(tmp_path):
    """A synthesised VARIANT without doc picks up purpose from uxStrings."""
    gen = _gen(tmp_path)

    variant = make_type(
        {
            'c': c.VARIANT,
            'uid': 'gws.ext.object.layer',
            'name': 'gws.ext.object.layer',
            'doc': '',
            'tMembers': {'qgis': 'gws.plugin.qgis.layer'},
        }
    )
    gen.typeDict[variant.uid] = variant

    gen.uxStrings = {
        'en': {
            'gws.ext.object.layer': {
                'purpose': 'A map layer rendered by some backend.',
            }
        }
    }

    strings.apply_ux_to_variants(gen)
    assert variant.doc == 'A map layer rendered by some backend.'


def test_apply_ux_to_variants_does_not_overwrite_existing_doc(tmp_path):
    gen = _gen(tmp_path)
    variant = make_type(
        {
            'c': c.VARIANT,
            'uid': 'gws.ext.object.layer',
            'name': 'gws.ext.object.layer',
            'doc': 'Existing doc.',
        }
    )
    gen.typeDict[variant.uid] = variant
    gen.uxStrings = {'en': {'gws.ext.object.layer': {'purpose': 'New purpose.'}}}

    strings.apply_ux_to_variants(gen)
    assert variant.doc == 'Existing doc.'


def test_configref_ux_block_renders_fields(tmp_path):
    """The configref renderer turns uxStrings entries into a markdown block."""
    from gws.spec.generator import configref

    gen = _gen(tmp_path)
    gen.uxStrings = {
        'de': {
            'gws.plugin.foo.Config': {
                'label': 'Foo-Provider',
                'purpose': 'Verbindet das Foo-System.',
                'complexity': 'intermediate',
            }
        }
    }

    creator = configref._Creator(gen, 'de')
    block = creator.ux_block('gws.plugin.foo.Config')
    assert '**Label:** Foo-Provider' in block
    assert '**Zweck:** Verbindet das Foo-System.' in block
    assert '**Komplexität:** intermediate' in block

    # No entry -> empty string.
    assert creator.ux_block('gws.plugin.foo.Config.host') == ''


def test_configref_ux_block_falls_back_to_english(tmp_path):
    from gws.spec.generator import configref

    gen = _gen(tmp_path)
    gen.uxStrings = {
        'en': {'gws.plugin.foo.Config': {'purpose': 'English only.'}},
        'de': {},
    }

    creator = configref._Creator(gen, 'de')
    block = creator.ux_block('gws.plugin.foo.Config')
    assert 'English only.' in block
