"""OWS operations of remote services."""

from typing import Optional

import gws
import gws.lib.mime


class Config(gws.Config):
    """OWS operation that overrides or adds to those in the service capabilities."""

    verb: gws.OwsVerb
    """OWS request type."""
    formats: Optional[list[str]]
    """MIME types supported by the operation."""
    params: Optional[dict]
    """Extra parameters added to GET requests for this operation."""
    postUrl: Optional[gws.Url]
    """URL for POST requests."""
    url: Optional[gws.Url]
    """URL for GET requests."""


def create(
    verb: gws.OwsVerb,
    op_from_caps: Optional[gws.OwsOperation] = None,
    cfgs: Optional[list[Config]] = None,
) -> gws.OwsOperation:
    """Create an operation from the capabilities and the configuration.

    All fields are initialized. Attributes of the operations with the same verb in the
    configuration replace those of ``op_from_caps``. The default MIME type is the first
    configured one, if ``formats`` are configured, otherwise selected from the supported ones.

    Args:
        verb: Request type.
        op_from_caps: Operation read from the capabilities, with MIME types as spelled by the service.
        cfgs: Operation configurations.

    Returns:
        The operation.
    """

    co = op_from_caps or gws.OwsOperation()
    op = gws.OwsOperation(
        allowedParameters=co.allowedParameters or {},
        constraints=co.constraints or {},
        mimeTypeMap=dict(co.mimeTypeMap or {}),
        mimeTypes=list(co.mimeTypes or []),
        params=co.params or {},
        postUrl=co.postUrl or '',
        url=co.url or '',
        verb=verb,
    )

    if op.mimeTypes:
        _set_mime_types(op, op.mimeTypes)

    has_formats_config = False

    if cfgs:
        for c in cfgs:
            if c.get('verb') != verb:
                continue
            p = c.get('params')
            if p is not None:
                op.params = p
            p = c.get('postUrl')
            if p is not None:
                op.postUrl = p
            p = c.get('url')
            if p is not None:
                op.url = p
            p = c.get('formats')
            if p is not None:
                _set_mime_types(op, p)
                has_formats_config = True

    if has_formats_config and op.mimeTypes:
        op.defaultMimeType = op.mimeTypes[0]
    else:
        op.defaultMimeType = _default_mime_type(op)

    return op


def _set_mime_types(op: gws.OwsOperation, values: list[str]):
    """Set the MIME types of an operation from the values a service offers.

    ``mimeTypes`` gets the values normalized with ``gws.lib.mime.get`` (aliases resolved), ``mimeTypeMap``
    maps those spelled differently by the service to the service spelling. Unknown MIME types are ignored.

    Args:
        op: Operation.
        values: MIME types as spelled by the service.
    """

    op.mimeTypes = []
    op.mimeTypeMap = {}
    for s in values:
        mt = gws.lib.mime.get(s)
        if not mt:
            gws.log.warning(f'{op.verb}: unknown MIME type {s!r} ignored')
            continue
        if mt in op.mimeTypes:
            continue
        op.mimeTypes.append(mt)
        if s != mt:
            op.mimeTypeMap[mt] = s


_PREFERRED_MIME_TYPES = {
    'image': [
        gws.lib.mime.PNG8,
        gws.lib.mime.PNG24,
        gws.lib.mime.PNG,
        gws.lib.mime.JPEG,
    ],
    'text': [
        gws.lib.mime.GML3,
        gws.lib.mime.GML32,
        gws.lib.mime.GML2,
        gws.lib.mime.GML,
        gws.lib.mime.XML,
    ],
}

_IMAGE_VERBS = {
    gws.OwsVerb.GetLegendGraphic,
    gws.OwsVerb.GetMap,
    gws.OwsVerb.GetTile,
}


def _default_mime_type(op: gws.OwsOperation) -> str:
    """Pick the default MIME type of an operation.

    The first preferred MIME type the operation supports is selected. Otherwise, the first supported
    MIME type with the same base type as a preferred one (``image/png; mode=16bit`` for ``image/png``),
    otherwise the first supported one. Without supported MIME types, the first preferred base type is used.
    """

    best_mts = _PREFERRED_MIME_TYPES['image' if op.verb in _IMAGE_VERBS else 'text']

    if not op.mimeTypes:
        return gws.lib.mime.get_base(best_mts[0])

    for mt in best_mts:
        if mt in op.mimeTypes:
            return mt

    for mt in op.mimeTypes:
        if gws.lib.mime.get_base(mt) in best_mts:
            return mt

    return op.mimeTypes[0]
