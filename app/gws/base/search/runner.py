import gws
import gws.types as t



class _LimitExceeded(Exception):
    pass


_DEFAULT_LIMIT = 10 * 1000


def run(req: gws.IWebRequest, args: gws.SearchArgs) -> t.List[gws.IFeature]:
    total_limit = args.limit or _DEFAULT_LIMIT
    used_layer_ids = set()
    features: t.List[gws.IFeature] = []

    dbg = [
        f'SEARCH ARGS',
        f"axis={args.axis}",
        f"bounds={args.bounds}",
        f"filter={args.filter}",
        f"keyword={args.keyword}",
        f"layers={[p.uid for p in args.layers]}",
        f"limit={args.limit}",
        f"params={args.params}",
        f"project={args.project.uid if args.project else None}",
        f"resolution={args.resolution}",
        f"shapes={[gws.props(p, req.user) for p in (args.shapes or [])]}",
        f"tolerance={args.tolerance}",
    ]

    gws.p(dbg)

    try:
        if args.layers:
            for layer in args.layers:
                used_layer_ids.add(layer.uid)
                for prov in layer.search_providers:
                    _run(req, args, prov, total_limit, features, layer=layer)

            for layer in args.layers:
                for anc_layer in layer.ancestors:
                    if anc_layer.uid not in used_layer_ids:
                        used_layer_ids.add(anc_layer.uid)
                        gws.log.debug(f'search ancestor={anc_layer.uid} for={layer.uid}')
                        for prov in anc_layer.search_providers:
                            _run(req, args, prov, total_limit, features, layer=anc_layer)

        if args.project:
            for prov in args.project.search_providers:
                _run(req, args, prov, total_limit, features, project=args.project)

    except _LimitExceeded:
        pass

    return features[:total_limit]


def _run(
        req: gws.IWebRequest, 
        args: gws.SearchArgs, 
        provider: gws.ISearchProvider, 
        total_limit, 
        features,
        layer: t.Optional[gws.ILayer] = None, 
        project: t.Optional[gws.IProject] = None, 
):
    args.limit = total_limit - len(features)
    if args.limit <= 0:
        raise _LimitExceeded()

    gws.log.debug(
        'SEARCH_BEGIN: prov=%r layer=%r limit=%d' % (gws.get(provider, 'uid'), gws.get(layer, 'uid'), args.limit))

    if not req.user.can_use(provider, context=layer or project):
        gws.log.debug('SEARCH_END: NO_ACCESS')
        return

    if not provider.can_run(args):
        gws.log.debug(f'SEARCH_END: N_A')
        return

    try:
        fs: t.List[gws.IFeature] = provider.run(args, layer) or []
    except:
        gws.log.exception('SEARCH_FAILED')
        return

    tt = provider.templates or (layer.templates if layer else None)
    dm = provider.data_model or (layer.data_model if layer else None)

    for f in fs:
        f.layer = layer
        f.category = provider.title or (layer.title if layer else '')
        f.templates = tt
        f.data_model = dm

    gws.log.debug('SEARCH_END, found=%r', len(fs))

    features.extend(fs)
