"""Search manager."""

from typing import Optional

import gws


class Object(gws.SearchManager):
    def run_search(self, search, user):
        results: list[gws.SearchResult] = []

        if search.layers:
            for layer in search.layers:
                for finder in layer.finders:
                    self._run_search(search, user, finder, layer, results)
                    if len(results) > search.limit:
                        return results[: search.limit]

        if search.project:
            for finder in search.project.finders:
                self._run_search(search, user, finder, None, results)
                if len(results) > search.limit:
                    return results[: search.limit]

        for finder in self.root.app.finders:
            self._run_search(search, user, finder, None, results)
            if len(results) > search.limit:
                return results[: search.limit]

        return results

    def _run_search(
        self,
        search: gws.SearchQuery,
        user: gws.User,
        finder: gws.Finder,
        layer: Optional[gws.Layer],
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
            features: list[gws.Feature] = finder.run(search, user, layer) or []
        except Exception:
            gws.log.exception('SEARCH_FAILED')
            return

        for feature in features:
            if finder.category:
                feature.category = finder.category
            elif finder.title:
                feature.category = finder.title
            elif not feature.category and layer and layer.title:
                feature.category = layer.title

            if search.bounds:
                feature.transform_to(search.bounds.crs)

            results.append(gws.SearchResult(feature=feature, layer=layer, finder=finder))
            if len(results) > search.limit:
                break

        gws.log.debug(f'SEARCH_END, found={len(features)}')
