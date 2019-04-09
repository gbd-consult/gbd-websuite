import gws
import gws.types as t


class _LimitExceeded(Exception):
    pass


def run(req, args: t.SearchArgs):
    # layer-provider-feature triples
    lpf = []

    total_limit = args.limit
    used_layers = set()

    try:
        if args.layers:
            for layer in args.layers:
                used_layers.add(layer.uid)
                for prov in layer.get_children('gws.ext.search.provider'):
                    _run(req, layer, prov, args, total_limit, lpf)

            for layer in args.layers:
                for par in _parents(layer):
                    if par.uid not in used_layers:
                        used_layers.add(par.uid)
                        gws.log.debug(f'search parent={par.uid} for={layer.uid}')
                        for prov in par.get_children('gws.ext.search.provider'):
                            _run(req, par, prov, args, total_limit, lpf)

        if args.project:
            for prov in args.project.get_children('gws.ext.search.provider'):
                _run(req, None, prov, args, total_limit, lpf)

    except _LimitExceeded:
        pass

    features = []

    for layer, prov, f in lpf[:total_limit]:
        f.provider = prov
        f.layer = layer
        f.transform(args.crs)
        features.append(f)

    return features


def _parents(layer):
    ps = []
    p = layer.parent
    while isinstance(p, t.LayerObject):
        ps.append(p)
        p = p.parent
    return ps


def _run(req, layer, prov: t.SearchProviderInterface, args: t.SearchArgs, total_limit, lpf):
    args.limit = total_limit - len(lpf)
    if args.limit <= 0:
        raise _LimitExceeded()

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
