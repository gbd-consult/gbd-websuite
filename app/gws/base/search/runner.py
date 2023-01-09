import gws
import gws.types as t


class _LimitExceeded(Exception):
    pass


_DEFAULT_LIMIT = 10 * 1000


class Result(gws.Data):
    feature: gws.IFeature
    layer: gws.ILayer
    finder: gws.IFinder


def run(search: gws.SearchArgs, user: gws.IUser) -> t.List[Result]:
    limit = search.limit or _DEFAULT_LIMIT
    used_layer_ids = set()
    results: t.List[Result] = []
    layers = search.layers or []

    for layer in layers:
        used_layer_ids.add(layer.uid)
        for finder in layer.finders:
            _run(search, user, finder, layer, results)
            if len(results) > limit:
                return results[:limit]

    for layer in search.layers:
        for ancestor in layer.ancestors():
            if ancestor.uid in used_layer_ids:
                continue
            used_layer_ids.add(ancestor.uid)
            gws.log.debug(f'search ancestor={ancestor.uid} for={layer.uid}')
            for finder in ancestor.finders:
                _run(search, user, finder, ancestor, results)
                if len(results) > limit:
                    return results[:limit]

    return results


def _run(
        search: gws.SearchArgs,
        user: gws.IUser,
        finder: gws.IFinder,
        layer: gws.ILayer,
        results,
):
    gws.log.debug(f'SEARCH_BEGIN: finder={finder.uid!r} layer={layer.uid!r}')

    if not user.can_use(finder, layer):
        gws.log.debug('SEARCH_END: no access')
        return

    if not finder.can_run(search, user):
        gws.log.debug(f'SEARCH_END: cannot run')
        return

    try:
        features: t.List[gws.IFeature] = finder.run(search, user, layer) or []
    except:
        gws.log.exception('SEARCH_FAILED')
        return

    for feature in features:
        results.append(Result(
            feature=feature,
            layer=layer,
            finder=finder,
        ))

    gws.log.debug(f'SEARCH_END, found={len(features)}')
