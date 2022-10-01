import gws
import gws.gis.crs
import gws.gis.source
import gws.lib.net
import gws.lib.mime
import gws.gis.ows
import gws.types as t


class OperationConfig(gws.Config):
    formats: t.Optional[t.List[str]]
    postUrl: t.Optional[gws.Url]
    url: gws.Url
    verb: gws.OwsVerb


class ProviderConfig(gws.Config):
    capsCacheMaxAge: gws.Duration = '1d'  #: max cache age for capabilities documents
    forceCrs: t.Optional[gws.CrsName]  #: use this CRS for requests
    maxRequests: int = 0  #: max concurrent requests to this source
    operations: t.Optional[t.List[OperationConfig]]
    url: gws.Url  #: service url


class Caps(gws.Data):
    metadata: gws.Metadata
    operations: t.List[gws.OwsOperation]
    sourceLayers: t.List[gws.SourceLayer]
    tileMatrixSets: t.List[gws.TileMatrixSet]
    version: str


class Provider(gws.Node, gws.IOwsProvider):
    def configure(self):
        self.forceCrs = gws.gis.crs.get(self.var('forceCrs'))
        self.sourceLayers = []
        self.url = self.var('url')
        self.version = ''

    def configure_operations(self, operations_from_caps):
        # add operations from the config, if any,
        # then add operations from the caps
        # so that configured ops take precedence

        self.operations = []

        for cfg in self.var('operations', default=[]):
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

        # check preferred formats

        def _best_format(op, best):
            # they support exactly what we want...
            for fmt in op.formats:
                if gws.lib.mime.get(fmt) == best:
                    return fmt
            # ...or they support anything at all...
            for fmt in op.formats:
                return fmt
            # ...otherwise, no preferred format
            return None

        image_ops = {
            gws.OwsVerb.GetLegendGraphic,
            gws.OwsVerb.GetMap,
            gws.OwsVerb.GetTile,
        }

        for op in self.operations:
            best = gws.lib.mime.PNG if op.verb in image_ops else gws.lib.mime.XML
            op.preferredFormat = _best_format(op, best)

    def operation(self, verb, method=None):
        for op in self.operations:
            if op.verb == verb:
                url = op.postUrl if method == gws.RequestMethod.POST else op.url
                if url:
                    return op

    def request_args_for_operation(self, op: gws.OwsOperation, method: gws.RequestMethod = None, params=None) -> gws.gis.ows.request.Args:
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
        url, params = gws.lib.net.extract_params(self.var('url'))
        op = gws.OwsOperation(
            formats=[gws.lib.mime.XML],
            url=url,
            params=params,
            verb=gws.OwsVerb.GetCapabilities,
        )
        args = self.request_args_for_operation(op)
        return gws.gis.ows.request.get_text(args, max_age=self.var('capsCacheMaxAge'))

    def find_features(self, args: gws.SearchArgs, source_layers: t.List[gws.SourceLayer]) -> t.List[gws.IFeature]:
        return []
