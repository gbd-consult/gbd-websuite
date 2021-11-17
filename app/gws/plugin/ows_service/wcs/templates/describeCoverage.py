import gws.plugin.ows_service.templatelib as tpl


def main(ARGS):
    def domain_set(lc):
        return (
            'gml:domainSet gml:RectifiedGrid',
            {'gml:id': lc.layer_pname, 'dimension': 2},
            ('gml:limits gml:GridEnvelope', ('gml:low', '0 0'), ('gml:high', '215323 125027')),
            ('gml:axisLabels', 'x y'),
            ('gml:origin gml:Point gml:pos', lc.extent[0], ' ', lc.extent[3]),
            ('gml:offsetVector', '0 1'),
            ('gml:offsetVector', '1 0'))

    def layer1(lc):
        yield 'label', lc.title
        yield 'name', lc.layer_pname
        yield tpl.lon_lat_envelope(lc)
        yield domain_set(lc)
        yield 'supportedCRSs', [('requestResponseCRSs', b.crs.epsg) for b in lc.bounds]
        yield 'supportedFormats formats', 'image/png'

    def doc1():
        yield {'xmlns': 'wcs'}
        for lc in ARGS.layer_caps_list:
            yield 'CoverageOffering', layer1(lc)

    def layer2(lc):
        yield {'gml:id': lc.layer_pname}
        yield 'gml:name', lc.layer_pname

        yield (
            'gml:boundedBy gml:Envelope',
            {
                'axisLabels': 'x y',
                'srsDimension': 2,
                'srsName': lc.bounds.crs.uri,
                'uomLabels': 'm m',
            },
            ('gml:lowerCorner', lc.extent[0], ' ', lc.extent[2]),
            ('gml:upperCorner', lc.extent[2], ' ', lc.extent[1]))

        yield domain_set(lc)

        yield (
            'ServiceParameters',
            ('CoverageSubtype', 'RectifiedGridCoverage'),
            ('nativeFormat', 'image/png'))

    def doc2():
        yield {'xmlns': 'wcs'}
        for lc in ARGS.layer_caps_list:
            yield 'CoverageDescription', layer2(lc)

    ##

    if ARGS.version.startswith('1'):
        return tpl.to_xml(ARGS, ('CoverageDescription', doc1()))

    if ARGS.version.startswith('2'):
        return tpl.to_xml(ARGS, ('CoverageDescriptions', doc2()))
