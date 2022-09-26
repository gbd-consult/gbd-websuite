import gws
import gws.gis.crs
import gws.gis.source
import gws.lib.metadata
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
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    url: gws.Url  #: service url


class Caps(gws.Data):
    metadata: dict
    operations: t.List[gws.OwsOperation]
    sourceLayers: t.List[gws.SourceLayer]
    version: str


class Provider(gws.Node, gws.IOwsProvider):
    def configure(self):
        self.forceCrs = gws.gis.crs.get(self.var('forceCrs'))
        self.sourceLayers = []
        self.url = self.var('url')
        self.version = ''

        # we need the Caps operation before we can go any further
        self.operations = [
            gws.OwsOperation(
                formats=[gws.lib.mime.XML],
                parameters={},
                url=self.var('url'),
                verb=gws.OwsVerb.GetCapabilities,
            )
        ]

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

    def operation_args(self, verb: gws.OwsVerb, method: gws.RequestMethod = None, params=None) -> gws.gis.ows.request.Args:
        op = self.operation(verb, method)
        if not op:
            raise gws.Error(f'operation not found: {verb!r}')

        rparams = {}

        if params:
            for name, val in params.items():
                name = name.upper()
                if name in op.parameters and val not in op.parameters[name]:
                    raise gws.Error(f'invalid parameter value {val!r} for {name!r}')
                rparams[name] = val

        for name, vals in op.parameters.items():
            if name not in rparams:
                rparams[name] = vals[0]

        return gws.gis.ows.request.Args(
            params=rparams,
            protocol=self.protocol,
            url=(op.postUrl if method == gws.RequestMethod.POST else op.url),
            verb=verb,
        )

    def get_capabilities(self):
        args = self.operation_args(gws.OwsVerb.GetCapabilities)
        return gws.gis.ows.request.get_text(
            args,
            max_age=self.var('capsCacheMaxAge'))

    def find_features(self, args: gws.SearchArgs, source_layers: t.List[gws.SourceLayer]) -> t.List[gws.IFeature]:
        return []
