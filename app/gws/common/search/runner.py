import gws
import gws.types as t


def run(req, args: t.SearchArgs):
    # layer-provider-feature triples
    lpf = []

    total_limit = args.limit

    if args.project:
        for prov in args.project.get_children('gws.ext.search.provider'):
            _run(req, None, prov, args, total_limit, lpf)

    if args.layers:
        for layer in args.layers:
            for prov in layer.get_children('gws.ext.search.provider'):
                _run(req, layer, prov, args, total_limit, lpf)

    features = []

    for layer, prov, f in lpf[:total_limit]:
        f.provider = prov
        f.layer = layer
        f.transform(args.crs)
        features.append(f)

    return features


def _run(req, layer, prov: t.SearchProviderInterface, args: t.SearchArgs, total_limit, lpf):
    args.limit = total_limit - len(lpf)
    if args.limit <= 0:
        return

    gws.log.debug(
        'SEARCH_BEGIN: prov=%r layer=%r limit=%d' % (gws.get(prov, 'uid'), gws.get(layer, 'uid'), args.limit))

    if not req.user.can_use(prov):
        gws.log.debug('SEARCH_END: NO_ACCESS')
        return

    if not prov.can_run(args):
        gws.log.debug(f'SEARCH_END: N_A')
        return

    try:
        res = prov.run(layer, args) or []
    except Exception:
        gws.log.exception()
        gws.log.debug('SEARCH_FAILED')
        return

    for f in res:
        lpf.append((layer, prov, f))

    gws.log.debug('SEARCH_END, found=%r', len(res))
    return res
