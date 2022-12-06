import gws
import gws.types as t


class _LimitExceeded(Exception):
    pass


_DEFAULT_LIMIT = 10 * 1000


def run(req: gws.IWebRequester, args: gws.SearchArgs) -> t.List[gws.IFeature]:
    total_limit = args.limit or _DEFAULT_LIMIT
    used_layer_ids = set()
    features: t.List[gws.IFeature] = []

    try:
        for layer in args.layers:
            used_layer_ids.add(layer.uid)
            for finder in layer.searchMgr.finders:
                _run(req, args, finder, total_limit, features, layer=layer)

        for layer in args.layers:
            for ancestor in layer.ancestors():
                if ancestor.uid in used_layer_ids:
                    continue
                used_layer_ids.add(ancestor.uid)
                gws.log.debug(f'search ancestor={ancestor.uid} for={layer.uid}')
                for finder in ancestor.searchMgr.finders:
                    _run(req, args, finder, total_limit, features, layer=ancestor)

    except _LimitExceeded:
        pass

    return features[:total_limit]


def _run(
        req: gws.IWebRequester,
        args: gws.SearchArgs,
        finder: gws.IFinder,
        total_limit,
        features,
        layer: t.Optional[gws.ILayer] = None,
):
    args.limit = total_limit - len(features)
    if args.limit <= 0:
        raise _LimitExceeded()

    gws.log.debug(f'SEARCH_BEGIN: finder={finder.uid!r} layer={layer.uid!r} limit={args.limit}')

    if not req.user.can_use(finder, layer):
        gws.log.debug('SEARCH_END: no access')
        return

    if not finder.can_run(args):
        gws.log.debug(f'SEARCH_END: cannot run')
        return

    try:
        fs: t.List[gws.IFeature] = finder.run(args, layer) or []
    except:
        gws.log.exception('SEARCH_FAILED')
        return

    template_mgr = finder.templateMgr or layer.templateMgr

    for elem in args.featureElements:
        tpl = template_mgr.get_template_for(args.user, subject='feature.' + elem)
        if tpl:
            for f in fs:
                f.apply_template(tpl)

    gws.log.debug(f'SEARCH_END, found={len(fs)}')

    features.extend(fs)
