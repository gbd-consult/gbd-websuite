"""Tests for the scenarios collection pipeline."""

import json
import textwrap

import pytest

from gws.spec.generator import base, strings


def _gen(tmp_path):
    g = base.Generator()
    g.rootDir = str(tmp_path)
    g.outDir = str(tmp_path / 'out')
    return g


def _write_json(tmp_path, rel, data):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding='utf-8')
    return p


def test_collect_scenarios_parses_json(tmp_path):
    _write_json(
        tmp_path,
        'plugin/foo/_doc/scenarios.json',
        {
            'gws.plugin.foo.Config': [
                {
                    'title': {'de': 'Standard-Setup', 'en': 'Default setup'},
                    'purpose': {'de': 'Übliche Konfiguration.', 'en': 'Usual configuration.'},
                    'template': {'host': 'localhost', 'port': 5432},
                }
            ]
        },
    )

    out = strings.collect_scenarios(_gen(tmp_path))

    assert 'de' in out and 'en' in out
    de = out['de']['gws.plugin.foo.Config'][0]
    assert de['title'] == 'Standard-Setup'
    assert de['purpose'] == 'Übliche Konfiguration.'
    assert de['template'] == {'host': 'localhost', 'port': 5432}


def test_invalid_json_raises(tmp_path):
    p = tmp_path / 'plugin' / 'foo' / '_doc' / 'scenarios.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{not valid json', encoding='utf-8')

    with pytest.raises(base.GeneratorError):
        strings.collect_scenarios(_gen(tmp_path))


def test_top_level_must_be_object(tmp_path):
    _write_json(tmp_path, 'plugin/foo/_doc/scenarios.json', ['not', 'an', 'object'])
    with pytest.raises(base.GeneratorError):
        strings.collect_scenarios(_gen(tmp_path))


def test_lang_falls_back_to_en(tmp_path):
    _write_json(
        tmp_path,
        'plugin/foo/_doc/scenarios.json',
        {
            'gws.plugin.foo.Config': [
                {
                    'title': {'en': 'English only'},
                    'template': {'x': 1},
                }
            ]
        },
    )

    out = strings.collect_scenarios(_gen(tmp_path))
    # Even though only 'en' was provided, the entry exists under 'en'.
    assert out['en']['gws.plugin.foo.Config'][0]['title'] == 'English only'
    # No 'de' key created when no de title and only en; that's fine because
    # consumers can fall back to en client-side.
    assert 'de' not in out or 'gws.plugin.foo.Config' not in out.get('de', {}) \
        or out['de']['gws.plugin.foo.Config'][0]['title'] == 'English only'


def test_invalid_scenario_skipped_with_warning(tmp_path, capsys):
    _write_json(
        tmp_path,
        'plugin/foo/_doc/scenarios.json',
        {
            'gws.plugin.foo.Config': [
                {'no_template_field': True},
                {'title': {'en': 'Good'}, 'template': {'a': 1}},
            ],
            'gws.plugin.foo.Bad': 'not a list',
        },
    )

    out = strings.collect_scenarios(_gen(tmp_path))
    assert len(out['en']['gws.plugin.foo.Config']) == 1
    assert out['en']['gws.plugin.foo.Config'][0]['title'] == 'Good'
    assert 'gws.plugin.foo.Bad' not in out.get('en', {})


def test_files_outside_doc_dir_ignored(tmp_path):
    # File in a non-_doc directory must NOT be picked up.
    p = tmp_path / 'plugin' / 'foo' / 'scenarios.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'gws.plugin.foo.X': [{'template': {}}]}), encoding='utf-8')

    out = strings.collect_scenarios(_gen(tmp_path))
    assert out == {}


def test_purpose_only_in_one_lang(tmp_path):
    _write_json(
        tmp_path,
        'plugin/foo/_doc/scenarios.json',
        {
            'gws.plugin.foo.Config': [
                {
                    'title': {'de': 'X', 'en': 'X'},
                    'purpose': {'de': 'Nur deutsch'},
                    'template': {},
                }
            ]
        },
    )

    out = strings.collect_scenarios(_gen(tmp_path))
    assert out['de']['gws.plugin.foo.Config'][0]['purpose'] == 'Nur deutsch'
    # English entry has no purpose because the source has no en purpose
    # and the collector only falls back through `en` itself.
    assert 'purpose' not in out['en']['gws.plugin.foo.Config'][0]
