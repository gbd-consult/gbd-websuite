import gws.plugin.ows_service.templatelib as tpl


def main(ARGS):
    def feature(fc):
        pfx, _ = tpl.split_name(fc.qname)
        return (
            fc.qname,
            {'gml:id': fc.feature.uid},
            [
                (pfx + ':' + a.name, a.value)
                for a in fc.feature.attributes
            ],
            (pfx + ':geometry', fc.shape_element)
        )

    def collection():
        coll = ARGS.collection
        yield {
            'timeStamp': coll.time_stamp,
            'numberMatched': coll.num_matched,
            'numberReturned': coll.num_returned,
        }

        for fc in coll.caps:
            yield 'wfs:member', feature(fc)

    return tpl.to_xml(ARGS, ('wfs:FeatureCollection', collection()))
