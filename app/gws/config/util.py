"""Helpers for ``configure`` methods: create common children and find providers."""

from typing import Optional, cast

import gws
import gws.gis.source


def configure_templates_for(obj: gws.Node, extra: Optional[list] = None) -> bool:
    """Create the templates of an object.

    Templates are created from the ``templates`` config and from ``extra``,
    using the object's ``create_template`` method if it has one.
    The result is stored in ``obj.templates``.

    Args:
        obj: Object to configure.
        extra: Additional template configs, appended after the configured ones.

    Returns:
        ``True`` if at least one template was created.
    """
    fn = _create_fn(obj, 'create_template', gws.ext.object.template)
    obj.templates = []

    p = obj.cfg('templates')
    if p:
        obj.templates.extend(gws.u.compact(fn(c) for c in p))

    if extra:
        obj.templates.extend(gws.u.compact(fn(c) for c in extra))

    return len(obj.templates) > 0


def configure_models_for(obj: gws.Node, with_default=False) -> bool:
    """Create the models of an object.

    Models are created from the ``models`` config, using the object's
    ``create_model`` method if it has one. The result is stored in ``obj.models``.

    Args:
        obj: Object to configure.
        with_default: If no models are configured, create one model with an empty config.

    Returns:
        ``True`` if models are configured or a default model was created.
    """
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
    """Create the finders of an object.

    Finders are created from the ``finders`` config, using the object's
    ``create_finder`` method if it has one. The result is stored in ``obj.finders``.

    Args:
        obj: Object to configure.
        with_default: If no finders are configured, create one finder with an empty config.

    Returns:
        ``True`` if finders are configured or a default finder was created.
    """
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
    """Return the object's create method with the given name, or a function that creates a child of ``cls``."""
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
    """Select the source layers of an object.

    If ``sourceLayers`` is configured, it is used as a filter on ``layers``.
    Otherwise, the internal ``_defaultSourceLayers`` config is used as is, if set.
    Otherwise, ``layers`` are filtered by the given flags.
    The result is stored in ``obj.sourceLayers``.

    Args:
        obj: Object to configure.
        layers: Source layers to select from.
        is_group: Filter by the group flag.
        is_image: Filter by the image flag.
        is_queryable: Filter by the queryable flag.
        is_visible: Filter by the visible flag.

    Returns:
        Always ``True``.
    """
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


def configure_provider_for(obj: gws.Node, cls: type) -> bool:
    """Set the provider of an object.

    If ``provider`` is configured, a shared provider object is created from it.
    Otherwise, the internal ``_defaultProvider`` config is used, if it is an instance of ``cls``.
    The result is stored in ``obj.provider``.

    Args:
        obj: Object to configure.
        cls: Provider class.

    Returns:
        ``True`` if a provider was set.

    Raises:
        ``gws.Error``: If no provider is found.
    """
    p = obj.cfg('provider')
    if p:
        obj.provider = obj.root.create_shared(cls, p)
        return True

    p = obj.cfg('_defaultProvider')
    if p and isinstance(p, cls):
        obj.provider = p
        return True

    raise gws.Error(f'no provider {cls!r} found for {obj!r}')


def configure_database_provider_for(obj: gws.Node, ext_type: Optional[str] = None) -> bool:
    """Set the database provider of an object.

    The provider is looked up by the ``dbUid`` config, then taken from the
    internal ``_defaultDb`` config, then found by the extension type.
    The result is stored in ``obj.db``.

    Args:
        obj: Object to configure.
        ext_type: Extension type of the provider, e.g. ``postgres``. Defaults to the object's ``extType``.

    Returns:
        ``True`` if a provider was set.

    Raises:
        ``gws.Error``: If ``dbUid`` is configured but not found, or if no provider is found.
    """
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
