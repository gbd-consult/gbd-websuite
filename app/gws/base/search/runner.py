import gws
import gws.types as t


def run(root: gws.IRoot, search: gws.SearchQuery, user: gws.IUser) -> list[gws.SearchResult]:
    results: list[gws.SearchResult] = []

    if search.layers:
        used_layer_ids = set()

        for layer in search.layers:
            used_layer_ids.add(layer.uid)
            for finder in layer.finders:
                _run(search, user, finder, layer, results)
                if len(results) > search.limit:
                    return results[:search.limit]

        for layer in search.layers:
            for ancestor in layer.ancestors():
                if ancestor.uid in used_layer_ids:
                    continue
                used_layer_ids.add(ancestor.uid)
                gws.log.debug(f'SEARCH_ANCESTOR {ancestor=} {layer=}')
                for finder in ancestor.finders:
                    _run(search, user, finder, ancestor, results)
                    if len(results) > search.limit:
                        return results[:search.limit]

    if search.project:
        for finder in search.project.finders:
            _run(search, user, finder, None, results)
            if len(results) > search.limit:
                return results[:search.limit]

    for finder in root.app.finders:
        _run(search, user, finder, None, results)
        if len(results) > search.limit:
            return results[:search.limit]

    return results


def _run(
        search: gws.SearchQuery,
        user: gws.IUser,
        finder: gws.IFinder,
        layer: t.Optional[gws.ILayer],
        results,
):
    gws.log.debug(f'SEARCH_BEGIN: {finder=} {layer=}')

    if not user.can_use(finder):
        gws.log.debug('SEARCH_END: no access')
        return

    if not finder.can_run(search, user):
        gws.log.debug(f'SEARCH_END: cannot run')
        return

    try:
        features: list[gws.IFeature] = finder.run(search, user, layer) or []
    except:
        gws.log.exception('SEARCH_FAILED')
        return

    for feature in features:
        if not feature.layerName and layer:
            feature.layerName = layer.title
        results.append(gws.SearchResult(feature=feature, layer=layer, finder=finder))
        if len(results) > search.limit:
            break

    gws.log.debug(f'SEARCH_END, found={len(features)}')
