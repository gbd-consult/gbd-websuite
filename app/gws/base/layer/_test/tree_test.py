import gws
import gws.gis.source
import gws.base.layer.tree as tree


def _tca(leaf_cfg, auto_layers=None):
    return tree._TreeConfigArgs(
        exclude_slf=None,
        flatten_config=None,
        auto_layers=auto_layers or [],
        create_leaf_layer_config=lambda sls: dict(leaf_cfg),
    )


def _source_layer(**kwargs):
    return gws.SourceLayer(name='layer_1', title='title_1', isVisible=True, isExpanded=False, isGroup=False, **kwargs)


def test_opacity_from_source_layer():
    cfg = tree._config(_tca({'type': 'type_1'}), _source_layer(opacity=0.3), 0)
    assert cfg['opacity'] == 0.3


def test_opacity_default():
    cfg = tree._config(_tca({'type': 'type_1'}), _source_layer(), 0)
    assert cfg['opacity'] == 1


def test_leaf_config_overrides_opacity():
    cfg = tree._config(_tca({'type': 'type_1', 'opacity': 1}), _source_layer(opacity=0.3), 0)
    assert cfg['opacity'] == 1


def test_leaf_config_overrides_title():
    cfg = tree._config(_tca({'type': 'type_1', 'title': 'title_2'}), _source_layer(), 0)
    assert cfg['title'] == 'title_2'


def test_leaf_config_keeps_defaults():
    cfg = tree._config(_tca({'type': 'type_1'}), _source_layer(), 0)
    assert cfg['type'] == 'type_1'
    assert cfg['title'] == 'title_1'
    assert cfg['clientOptions'] == {'hidden': False, 'expanded': False}


def test_auto_layers_override_leaf_opacity():
    auto = gws.Data(applyTo=gws.gis.source.LayerFilter(names=['layer_1']), config={'opacity': 0.7})
    cfg = tree._config(_tca({'type': 'type_1', 'opacity': 1}, [auto]), _source_layer(opacity=0.3), 0)
    assert cfg['opacity'] == 0.7
