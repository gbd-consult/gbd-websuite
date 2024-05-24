"""Common configuration utilities."""

from typing import Optional, cast

import gws
import gws.gis.source


def configure_templates_for(obj: gws.Node, extra: Optional[list] = None) -> bool:
    fn = _create_fn(obj, 'create_template', gws.ext.object.template)
    obj.templates = []

    p = obj.cfg('templates')
    if p:
        obj.templates.extend(gws.u.compact(fn(c) for c in p))

    if extra:
        obj.templates.extend(gws.u.compact(fn(c) for c in extra))

    return len(obj.templates) > 0


def configure_models_for(obj: gws.Node, with_default=False) -> bool:
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


def configure_finders_for(obj: gws.Node, with_default=False) -> bool:
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


def configure_source_layers_for(
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


def configure_service_provider_for(obj: gws.Node, cls: type) -> bool:
    p = obj.cfg('provider')
    if p:
        obj.serviceProvider = obj.root.create_shared(cls, p)
        return True

    p = obj.cfg('_defaultProvider')
    if p and isinstance(p, cls):
        obj.serviceProvider = p
        return True

    raise gws.Error(f'no provider {cls!r} found for {obj!r}')


def configure_database_provider_for(obj: gws.Node, ext_type: Optional[str] = None) -> bool:
    mgr = obj.root.app.databaseMgr

    uid = obj.cfg('dbUid')
    if uid:
        p = mgr.find_provider(uid=uid)
        if p:
            obj.db = p
            return True
        raise gws.Error(f'database provider {uid!r} not found')

    p = obj.cfg('_defaultDb')
    if p:
        obj.db = p
        return True

    ext_type = ext_type or obj.extType
    if ext_type:
        p = mgr.find_provider(ext_type=ext_type)
        if p:
            obj.db = p
            return True

    raise gws.Error(f'no database providers of type {ext_type!r} configured')
