"""Base OWS service provider."""

from typing import Optional

import gws
import gws.lib.crs
import gws.gis.source
import gws.lib.net
import gws.lib.mime
import gws.lib.text

from . import request, operation

_MIME_TYPE_PARAMS = {'FORMAT', 'INFO_FORMAT', 'OUTPUTFORMAT'}


class AuthorizationConfig(gws.Config):
    """Credentials for accessing the service."""

    type: str
    """Authorization type ("basic" or "token")."""
    username: str = ''
    """User name."""
    password: str = ''
    """Password."""
    header: str = ''
    """HTTP header that carries the token."""
    prefix: str = ''
    """Prefix expected before the token in the header value."""
    token: str = ''
    """Authorization token."""


class Config(gws.Config):
    """OWS provider configuration."""

    capsCacheMaxAge: gws.Duration = '1d'
    """How long a downloaded capabilities document is cached."""
    forceCrs: Optional[gws.CrsName]
    """Use this CRS for requests."""
    alwaysXY: bool = False
    """Force XY axis order for CRSs with lat/lon axis order."""
    maxRequests: int = 0
    """Max. concurrent requests to this source."""
    operations: Optional[list[operation.Config]]
    """Override operations reported in capabilities."""
    authorization: Optional[AuthorizationConfig]
    """Service authorization options."""
    url: gws.Url
    """Service URL."""


class Object(gws.OwsServiceProvider):
    """Base OWS service provider.

    Holds the service URL, version, operations and source layers of a remote
    OWS service, and prepares request arguments for its operations. Provides
    subclasses with capabilities download and caching, merging of configured
    and reported operations and format selection. Protocol-specific subclasses
    set ``protocol`` and parse the capabilities.
    """

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
        """Set up the provider operations.

        Operations from the configuration replace those from the capabilities
        with the same verb; attributes missing in the configuration are taken from
        the capabilities (see ``operation.create``).

        Args:
            operations_from_caps: Operations read from the capabilities.
        """

        ops = {}
        cfgs = self.cfg('operations')

        for op in operations_from_caps:
            ops[op.verb] = operation.create(op.verb, op, cfgs)

        if cfgs:
            for c in cfgs:
                if c.get('verb') not in ops:
                    ops[c.get('verb')] = operation.create(c.get('verb'), None, cfgs)

        self.operations = list(ops.values())

    def cache_hash(self):
        """Compute a hash of the provider settings, used to build cache keys.

        Returns:
            A hash of the URL, the forced CRS and the axis order setting.
        """

        return gws.u.sha256(
            [
                self.url,
                self.forceCrs.srid if self.forceCrs else '',
                self.alwaysXY,
            ]
        )

    def get_operation(self, verb, method=None):
        for op in self.operations:
            if op.verb == verb:
                url = op.postUrl if method == gws.RequestMethod.POST else op.url
                if url:
                    return op

    def prepare_operation(self, op: gws.OwsOperation, method: gws.RequestMethod = None, params=None) -> request.Args:
        """Prepare the request arguments for an operation.

        For GET requests, the operation parameters are included. Parameter names
        are upper-cased and checked against the operation's allowed values. MIME types
        in ``FORMAT``, ``INFO_FORMAT`` and ``OUTPUTFORMAT`` are sent in the spelling of the service.
        Adds a basic ``Authorization`` header if configured.

        Args:
            op: Operation.
            method: Request method, ``GET`` by default.
            params: Request parameters.

        Returns:
            Request arguments, to be passed to ``request.get``.

        Raises:
            ``gws.Error``: If a parameter value is not allowed by the operation.
        """

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

        if params:
            for name, val in params.items():
                name = name.upper()
                if name in _MIME_TYPE_PARAMS and val in op.mimeTypeMap:
                    val = op.mimeTypeMap[val]
                if name in op.allowedParameters and val not in op.allowedParameters[name]:
                    raise gws.Error(f'invalid parameter value {val!r} for {name!r}')
                args.params[name] = val

        if self.authorization and self.authorization.type == 'basic':
            u = self.authorization.username or ''
            p = self.authorization.password or ''
            b = gws.lib.text.to_base64(f'{u}:{p}')
            args.headers['Authorization'] = 'Basic ' + b

        if self.authorization and self.authorization.type == 'token':
            h = self.authorization.header or ''
            p = self.authorization.prefix or ''
            t = self.authorization.token or ''
            args.headers[h] = f'{p} {t}'

        return args

    def get_capabilities(self):
        """Download the capabilities document of the service.

        The request goes to the service URL, unless the configuration has a ``GetCapabilities``
        operation with its own attributes. The document is cached for ``capsCacheMaxAge``.

        Returns:
            The capabilities XML text.

        Raises:
            ``gws.ExternalServiceError``: If the request fails.
        """

        url, params = gws.lib.net.extract_params(self.url)
        op = operation.create(
            gws.OwsVerb.GetCapabilities,
            gws.OwsOperation(mimeTypes=[gws.lib.mime.XML], params=params, url=url),
            self.cfg('operations'),
        )
        args = self.prepare_operation(op)
        return gws.u.get_cached_object(
            f'ows_caps_{gws.u.sha256([args.url, args.params])}',
            self.cfg('capsCacheMaxAge'),
            lambda: request.get_text(args),
        )


##
