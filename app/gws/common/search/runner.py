import gws
import gws.types as t

from . import provider


class _LimitExceeded(Exception):
    pass

_DEFAULT_LIMIT = 10 * 1000


def run(req, args: t.SearchArgs) -> t.List[t.IFeature]:
    total_limit = args.limit or _DEFAULT_LIMIT
    used_layer_ids = set()
    features: t.List[t.IFeature] = []
    prov: provider.Object

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
        f"shapes={[p.props for p in (args.shapes or [])]}",
        f"tolerance={args.tolerance}",
    ]

    gws.p(dbg)

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


def _parents(layer: t.ILayer) -> t.List[t.ILayer]:
    ps = []
    p = layer.parent
    while p.is_a('gws.ext.layer'):
        ps.append(t.cast(t.ILayer, p))
        p = p.parent
    return ps


def _run(req, layer: t.Optional[t.ILayer], prov: provider.Object, args: t.SearchArgs, total_limit, features):
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
        fs: t.List[t.IFeature] = prov.run(layer, args) or []
    except Exception:
        gws.log.exception()
        gws.log.debug('SEARCH_FAILED')
        return

    tt = prov.templates or (layer.templates if layer else None)
    dm = prov.data_model or (layer.data_model if layer else None)

    for f in fs:
        f.layer = layer
        f.templates = tt
        f.data_model = dm

    gws.log.debug('SEARCH_END, found=%r', len(fs))

    features.extend(fs)
