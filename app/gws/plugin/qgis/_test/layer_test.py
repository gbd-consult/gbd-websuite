"""Tests for the withCache defaults and propagation of the qgis layers."""

import gws
import gws.gis.source
import gws.base.layer.tree as tree
import gws.test.util as u
import gws.plugin.qgis.layer as layer


class _Provider:
    def create_leaf_layer_config(self, source_layers):
        return {'type': 'type_1'}


def _leaf_config_fn(monkeypatch, config):
    la = layer.Object()
    la.config = gws.Config(config)
    la.provider = _Provider()
    la.provider.sourceLayers = []
    captured = []
    monkeypatch.setattr(tree, 'configure_group_layers_for', lambda layer, sls, fn: captured.append(fn))
    la.configure_group()
    return captured[0]


def _source_layer():
    return gws.SourceLayer(name='layer_1', title='title_1', isVisible=True, isExpanded=False, isGroup=False)


def test_qgis_with_cache_default():
    cfg = u.gws_specs().read({'type': 'qgis'}, 'gws.plugin.qgis.layer.Config')
    assert cfg.withCache is False


def test_qgisflat_with_cache_default():
    cfg = u.gws_specs().read({'type': 'qgisflat'}, 'gws.plugin.qgis.flatlayer.Config')
    assert cfg.withCache is False


def test_with_cache_passed_to_leaves(monkeypatch):
    cfg = _leaf_config_fn(monkeypatch, {'withCache': True})([_source_layer()])
    assert cfg['withCache'] is True


def test_without_cache_not_passed_to_leaves(monkeypatch):
    cfg = _leaf_config_fn(monkeypatch, {'withCache': False})([_source_layer()])
    assert 'withCache' not in cfg


def test_auto_layers_override_with_cache(monkeypatch):
    fn = _leaf_config_fn(monkeypatch, {'withCache': True})
    auto = gws.Data(applyTo=gws.gis.source.LayerFilter(names=['layer_1']), config={'withCache': False})
    tca = tree._TreeConfigArgs(
        exclude_slf=None,
        flatten_config=None,
        auto_layers=[auto],
        create_leaf_layer_config=fn,
    )
    cfg = tree._config(tca, _source_layer(), 0)
    assert cfg['withCache'] is False


def test_cache_passed_to_leaves_without_name(monkeypatch):
    cache = {'maxAge': 100, 'maxLevel': 10, 'name': 'name_1'}
    cfg = _leaf_config_fn(monkeypatch, {'cache': cache})([_source_layer()])
    assert cfg['cache'] == {'maxAge': 100, 'maxLevel': 10, 'name': ''}
    assert cache['name'] == 'name_1'


def test_no_cache_not_passed_to_leaves(monkeypatch):
    cfg = _leaf_config_fn(monkeypatch, {})([_source_layer()])
    assert 'cache' not in cfg


##


def _configured_layer(config):
    la = layer.Object()
    la.config = gws.Config(config)
    warnings = []
    la.root = gws.Data(config_warning=warnings.append)
    la.configure()
    return la, warnings


def test_with_composite_render():
    la, warnings = _configured_layer({'withCompositeRender': True})
    assert la.withCompositeRender is True
    assert la.canRenderBox is True
    assert la.canRenderTile is True
    assert warnings == []


def test_without_composite_render():
    la, warnings = _configured_layer({})
    assert la.withCompositeRender is False
    assert warnings == []


def test_composite_render_deprecated():
    la, warnings = _configured_layer({'compositeRender': True})
    assert la.withCompositeRender is True
    assert la.canRenderBox is True
    assert len(warnings) == 1
    assert 'compositeRender' in warnings[0]


def test_composite_render_deprecated_false():
    la, warnings = _configured_layer({'compositeRender': False})
    assert la.withCompositeRender is False
    assert len(warnings) == 1
