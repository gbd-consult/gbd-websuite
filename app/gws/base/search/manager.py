"""Search manager."""

import gws
import gws.types as t


class Object(gws.SearchManager):
    def run_search(self, search, user):
        results: list[gws.SearchResult] = []

        if search.layers:
            for layer in search.layers:
                for finder in layer.finders:
                    self._run(search, user, finder, layer, results)
                    if len(results) > search.limit:
                        return results[:search.limit]

            # # searching ancestors is unnecessary, the client sends the whole tree path anyways
            # for layer in search.layers:
            #     for ancestor in layer.ancestors():
            #         if ancestor.uid in used_layer_ids:
            #             continue
            #         used_layer_ids.add(ancestor.uid)
            #         gws.log.debug(f'SEARCH_ANCESTOR {ancestor=} {layer=}')
            #         for finder in ancestor.finders:
            #             _run(search, user, finder, ancestor, results)
            #             if len(results) > search.limit:
            #                 return results[:search.limit]

        if search.project:
            for finder in search.project.finders:
                self._run(search, user, finder, None, results)
                if len(results) > search.limit:
                    return results[:search.limit]

        for finder in self.root.app.finders:
            self._run(search, user, finder, None, results)
            if len(results) > search.limit:
                return results[:search.limit]

        return results

    def _run(
            self,
            search: gws.SearchQuery,
            user: gws.User,
            finder: gws.Finder,
            layer: t.Optional[gws.Layer],
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
        except:
            gws.log.exception('SEARCH_FAILED')
            return

        for feature in features:
            if finder.title:
                feature.category = finder.title
            elif not feature.category and layer and layer.title:
                feature.category = layer.title

            results.append(gws.SearchResult(feature=feature, layer=layer, finder=finder))
            if len(results) > search.limit:
                break

        gws.log.debug(f'SEARCH_END, found={len(features)}')
