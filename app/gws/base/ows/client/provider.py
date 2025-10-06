from typing import Optional
import base64

import gws
import gws.lib.crs
import gws.gis.source
import gws.lib.net
import gws.lib.mime

from . import request

_IMAGE_VERBS = {
    gws.OwsVerb.GetLegendGraphic,
    gws.OwsVerb.GetMap,
    gws.OwsVerb.GetTile,
}

_PREFER_IMAGE_MIME = [gws.lib.mime.PNG, gws.lib.mime.JPEG]
_PREFER_XML_MIME = [gws.lib.mime.GML3, gws.lib.mime.GML, gws.lib.mime.XML]


class OperationConfig(gws.Config):
    """Custom OWS operation."""
    verb: gws.OwsVerb
    """OWS verb."""
    formats: Optional[list[str]]
    """Supported formats."""
    params: Optional[dict]
    """Operation parameters."""
    postUrl: Optional[gws.Url]
    """URL for POST requests."""
    url: Optional[gws.Url]
    """URL for GET requests."""


class AuthorizationConfig(gws.Config):
    """Service authorization."""
    type: str
    """Authorization type (only "basic" is supported)."""
    username: str = ''
    """User name."""
    password: str = ''
    """Password."""


class Config(gws.Config):
    """OWS provider configuration."""

    capsCacheMaxAge: gws.Duration = '1d'
    """Max cache age for capabilities documents."""
    forceCrs: Optional[gws.CrsName]
    """Use this CRS for requests."""
    alwaysXY: bool = False
    """Force XY orientation for lat/lon projections."""
    maxRequests: int = 0
    """Max concurrent requests to this source."""
    operations: Optional[list[OperationConfig]]
    """Override operations reported in capabilities."""
    authorization: Optional[AuthorizationConfig]
    """Service authorization."""
    url: gws.Url
    """Service url."""


class Object(gws.OwsProvider):
    def configure(self):
        self.alwaysXY = self.cfg('alwaysXY', default=False)
        self.forceCrs = gws.lib.crs.get(self.cfg('forceCrs'))
        self.maxRequests = self.cfg('maxRequests')
        self.operations = []
        self.sourceLayers = []
        self.url = self.cfg('url')
        self.version = ''

        p = self.cfg('authorization')
        self.authorization = gws.OwsAuthorization(p) if p else None

    def configure_operations(self, operations_from_caps):
        d = {}

        for op in operations_from_caps:
            d[op.verb] = op

        for cfg in (self.cfg('operations') or []):
            # add an operation from the config, borrowing missing attributes from a caps op
            verb = cfg.get('verb')
            caps_op = d.get(verb, {})
            d[verb] = gws.OwsOperation(
                verb=verb,
                formats=gws.u.first_not_none(cfg.get('formats'), caps_op.get('formats'), []),
                params=gws.u.first_not_none(cfg.get('params'), caps_op.get('params'), {}),
                postUrl=gws.u.first_not_none(cfg.get('postUrl'), caps_op.get('postUrl'), ''),
                url=gws.u.first_not_none(cfg.get('url'), caps_op.get('url'), '')
            )

        self.operations = list(d.values())

        for op in self.operations:
            op.preferredFormat = self._preferred_format(op)

    def _preferred_format(self, op: gws.OwsOperation) -> Optional[str]:
        prefer_fmts = _PREFER_IMAGE_MIME if op.verb in _IMAGE_VERBS else _PREFER_XML_MIME

        if not op.formats:
            return prefer_fmts[0]

        # select the "best" (leftmost in the preferred list) format

        best_pos = 999
        best_fmt = ''

        for fmt in op.formats:
            canon_fmt = gws.lib.mime.get(fmt)
            if not canon_fmt or canon_fmt not in prefer_fmts:
                continue
            pos = prefer_fmts.index(canon_fmt)
            if pos < best_pos:
                best_pos = pos
                best_fmt = fmt

        return best_fmt or op.formats[0]

    def get_operation(self, verb, method=None):
        for op in self.operations:
            if op.verb == verb:
                url = op.postUrl if method == gws.RequestMethod.POST else op.url
                if url:
                    return op

    def prepare_operation(self, op: gws.OwsOperation, method: gws.RequestMethod = None, params=None) -> request.Args:
        args = request.Args(
            method=method or gws.RequestMethod.GET,
            headers={},
            params={},
            protocol=self.protocol,
            verb=op.verb,
            version=self.version,
        )

        if args.method == gws.RequestMethod.GET:
            args.params.update(op.params)

        args.url = op.url
        if args.method == gws.RequestMethod.POST:
            args.url = op.postUrl

        allowed = op.allowedParameters or {}

        if params:
            for name, val in params.items():
                name = name.upper()
                if name in allowed and val not in allowed[name]:
                    raise gws.Error(f'invalid parameter value {val!r} for {name!r}')
                args.params[name] = val

        if self.authorization and self.authorization.type == 'basic':
            b = base64.encodebytes(
                gws.u.to_bytes(self.authorization.username) + b':' + gws.u.to_bytes(self.authorization.password))
            args.headers['Authorization'] = 'Basic ' + gws.u.to_str(b).strip()

        return args

    def get_capabilities(self):
        url, params = gws.lib.net.extract_params(self.url)
        op = gws.OwsOperation(
            formats=[gws.lib.mime.XML],
            url=url,
            params=params,
            verb=gws.OwsVerb.GetCapabilities,
        )
        args = self.prepare_operation(op)
        return request.get_text(args, max_age=self.cfg('capsCacheMaxAge'))
