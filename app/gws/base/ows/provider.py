import gws
import gws.gis.crs
import gws.gis.source
import gws.lib.net
import gws.lib.mime
import gws.gis.ows
import gws.types as t

_IMAGE_VERBS = {
    gws.OwsVerb.GetLegendGraphic,
    gws.OwsVerb.GetMap,
    gws.OwsVerb.GetTile,
}

_PREFER_IMAGE_MIME = {gws.lib.mime.PNG}

_PREFER_XML_MIME = {gws.lib.mime.GML3, gws.lib.mime.GML, gws.lib.mime.XML}


class Object(gws.Node, gws.IOwsProvider):
    def configure(self):
        self.alwaysXY = self.cfg('alwaysXY', default=False)
        self.forceCrs = gws.gis.crs.get(self.cfg('forceCrs'))
        self.maxRequests = self.cfg('maxRequests')
        self.sourceLayers = []
        self.url = self.cfg('url')
        self.version = ''

    def configure_operations(self, operations_from_caps):
        # add operations from the config, if any,
        # then add operations from the caps
        # so that configured ops take precedence

        self.operations = []

        for cfg in self.cfg('operations', default=[]):
            self.operations.append(gws.OwsOperation(
                formats=cfg.get('formats', []),
                params={},
                postUrl=cfg.get('postUrl'),
                url=cfg.get('url'),
                verb=cfg.get('verb'),
            ))

        verbs = set(op.verb for op in self.operations)

        for op in operations_from_caps:
            if op.verb not in verbs:
                self.operations.append(op)

        for op in self.operations:
            op.preferredFormat = self._preferred_format(op)

    def _preferred_format(self, op: gws.OwsOperation) -> t.Optional[str]:
        mime = _PREFER_IMAGE_MIME if op.verb in _IMAGE_VERBS else _PREFER_XML_MIME
        for fmt in op.formats:
            if gws.lib.mime.get(fmt) in mime:
                return fmt
        for fmt in op.formats:
            return fmt

    def get_operation(self, verb, method=None):
        for op in self.operations:
            if op.verb == verb:
                url = op.postUrl if method == gws.RequestMethod.POST else op.url
                if url:
                    return op

    def prepare_operation(self, op: gws.OwsOperation, method: gws.RequestMethod = None, params=None) -> gws.gis.ows.request.Args:
        args = gws.gis.ows.request.Args(
            method=method or gws.RequestMethod.GET,
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

        for name, vals in allowed.items():
            if name not in args.params:
                args.params[name] = vals[0]

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
        return gws.gis.ows.request.get_text(args, max_age=self.cfg('capsCacheMaxAge'))
