import gws
import gws.types as t

from . import provider


class _LimitExceeded(Exception):
    pass


def run(req, args: t.SearchArguments) -> t.List[t.Feature]:
    total_limit = args.limit
    used_layer_ids = set()
    features: t.List[t.Feature] = []
    prov: provider.Object

    try:
        if args.layers:
            for layer in args.layers:
                used_layer_ids.add(layer.uid)
                for prov in layer.get_children('gws.ext.search.provider'):
                    _run(req, layer, prov, args, total_limit, features)

            for layer in args.layers:
                for par in _parents(layer):
                    if par.uid not in used_layer_ids:
                        used_layer_ids.add(par.uid)
                        gws.log.debug(f'search parent={par.uid} for={layer.uid}')
                        for prov in par.get_children('gws.ext.search.provider'):
                            _run(req, par, prov, args, total_limit, features)

        if args.project:
            for prov in args.project.get_children('gws.ext.search.provider'):
                _run(req, None, prov, args, total_limit, features)

    except _LimitExceeded:
        pass

    return features[:total_limit]


def _parents(layer):
    ps = []
    p = layer.parent
    while isinstance(p, t.LayerObject):
        ps.append(p)
        p = p.parent
    return ps


def _run(req, layer, prov: provider.Object, args: t.SearchArguments, total_limit, features):
    args.limit = total_limit - len(features)
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
        fs = prov.run(layer, args) or []
    except Exception:
        gws.log.exception()
        gws.log.debug('SEARCH_FAILED')
        return

    for f in fs:
        f.layer = layer
        f.convertor = layer or prov

    gws.log.debug('SEARCH_END, found=%r', len(fs))

    features.extend(fs)
