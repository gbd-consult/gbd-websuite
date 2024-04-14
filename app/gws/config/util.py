"""Common configuration utilities."""

import gws
import gws.gis.source
import gws.types as t


def configure_templates(obj: gws.Node, extra: t.Optional[list] = None) -> bool:
    fn = _create_fn(obj, 'create_template', gws.ext.object.template)
    obj.templates = []

    p = obj.cfg('templates')
    if p:
        obj.templates.extend(gws.u.compact(fn(c) for c in p))

    if extra:
        obj.templates.extend(gws.u.compact(fn(c) for c in extra))

    return len(obj.templates) > 0


def configure_models(obj: gws.Node, with_default=False) -> bool:
    fn = _create_fn(obj, 'create_model', gws.ext.object.model)
    obj.models = []

    p = obj.cfg('models')
    if p:
        obj.models = gws.u.compact(fn(c) for c in p)
        return True

    if with_default:
        obj.models = [fn(None)]
        return True

    return False


def configure_finders(obj: gws.Node, with_default=False) -> bool:
    fn = _create_fn(obj, 'create_finder', gws.ext.object.finder)
    obj.finders = []

    p = obj.cfg('finders')
    if p:
        obj.finders = gws.u.compact(fn(c) for c in p)
        return True

    if with_default:
        obj.finders = [fn(None)]
        return True

    return False


def _create_fn(obj, name: str, cls: type):
    fn = getattr(obj, name, None)
    if fn:
        return fn
    return lambda c: obj.create_child(cls, c)


def configure_source_layers(
        obj: gws.Node,
        layers: list[gws.SourceLayer],
        is_group: bool = None,
        is_image: bool = None,
        is_queryable: bool = None,
        is_visible: bool = None,
) -> bool:
    p = obj.cfg('sourceLayers')
    if p:
        obj.sourceLayers = gws.gis.source.filter_layers(layers, p)
        return True

    p = obj.cfg('_defaultSourceLayers')
    if p:
        obj.sourceLayers = p
        return True

    obj.sourceLayers = gws.gis.source.filter_layers(
        layers,
        is_group=is_group,
        is_image=is_image,
        is_queryable=is_queryable,
        is_visible=is_visible,
    )
    return True


def get_provider(cls: type, obj: gws.Node):
    p = obj.cfg('provider')
    if p:
        return obj.root.create_shared(cls, p)

    p = obj.cfg('_defaultProvider')
    if p and isinstance(p, cls):
        return p

    raise gws.Error(f'no provider found for {obj!r}')
