import gws.plugin.ows_service.csw.templates.iso_record as rec
import gws.plugin.ows_service.templatelib as tpl


def main(ARGS):
    def response():
        yield 'csw:SearchStatus', {'timestamp': ARGS.results.timestamp}
        yield (
            'csw:SearchResults', {
                'elementSet': 'full',
                'nextRecord': ARGS.results.next,
                'numberOfRecordsMatched': ARGS.results.count_total,
                'numberOfRecordsReturned': ARGS.results.count_return,
                'recordSchema': 'http://www.isotc211.org/2005/gmd',
            }, [rec.record(ARGS, md) for md in ARGS.records]
        )

    return tpl.soap_wrap(ARGS, (
        'csw:GetRecordsResponse',
        {'version': ARGS.version},
        response()
    ))
