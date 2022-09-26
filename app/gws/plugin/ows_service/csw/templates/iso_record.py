import gws.lib.xmlx as xmlx
import gws.plugin.ows_service.templatelib as tpl


def record(ARGS, md):
    ML_GMX_CODELISTS = 'http://standards.iso.org/iso/19139/resources/gmxCodelists.xml'

    def CODE(wrap, lst, value, text=None):
        return (
            f'gmd:{wrap} gmd:{lst}',
            {'codeList': ML_GMX_CODELISTS + '#' + lst, 'codeListValue': value},
            text or value
        )

    def DATE(d, typ):
        return (
            'gmd:date gmd:CI_Date',
            ('gmd:date gco:Date', d),
            CODE('dateType', 'CI_DateTypeCode', typ)
        )

    def LANGUAGE():
        return (
            'gmd:language gmd:LanguageCode',
            {'codeList': 'http://www.loc.gov/standards/iso639-2/', 'codeListValue': md.language3},
            md.languageName
        )

    def BOUNDING_BOX(ext):
        return (
            'gmd:EX_GeographicBoundingBox',
            ('gmd:westBoundLongitude gco:Decimal', ext[0]),
            ('gmd:eastBoundLongitude gco:Decimal', ext[2]),
            ('gmd:southBoundLatitude gco:Decimal', ext[1]),
            ('gmd:northBoundLatitude gco:Decimal', ext[3]),
        )

    def contact():
        yield (
            'gmd:CI_ResponsibleParty',
            ('gmd:organisationName gco:CharacterString', md.contactOrganization),
            ('gmd:positionName gco:CharacterString', md.contactPosition),
            (
                'gmd:contactInfo gmd:CI_Contact',
                (
                    'gmd:phone gmd:CI_Telephone',
                    ('gmd:voice gco:CharacterString', md.contactPhone),
                    ('gmd:facsimile gco:CharacterString', md.contactFax),
                ),
                (
                    'gmd:address gmd:CI_Address',
                    ('gmd:deliveryPoint gco:CharacterString', md.contactAddress),
                    ('gmd:city gco:CharacterString', md.contactCity),
                    ('gmd:administrativeArea gco:CharacterString', md.contactArea),
                    ('gmd:postalCode gco:CharacterString', md.contactZip),
                    ('gmd:country gco:CharacterString', md.contactCountry),
                    ('gmd:electronicMailAddress gco:CharacterString', md.contactEmail),
                ),
                ('gmd:onlineResource gmd:CI_OnlineResource gmd:linkage gmd:URL', md.contactUrl),
            ),
            CODE('role', 'CI_RoleCode', md.contactRole)
        )

    def identification():
        yield (
            'gmd:citation gmd:CI_Citation',
            ('gmd:title gco:CharacterString', md.title),
            DATE('md.dateCreated', 'publication'),
            DATE('md.dateUpdated', 'revision'),
            ('gmd:identifier gmd:MD_Identifier gmd:code gco:CharacterString', md.catalogCitationUid)
        )

        yield 'gmd:abstract gco:CharacterString', md.abstract

        yield 'gmd:pointOfContact', contact()

        if md.inspireSpatialScope:
            lst = 'http://inspire.ec.europa.eu/metadata-codelist/SpatialScope/'
            yield (
                'gmd:descriptiveKeywords gmd:MD_Keywords',
                (
                    'gmd:keyword gmx:Anchor',
                    {'xlink:href': lst + md.inspireSpatialScope},
                    md.inspireSpatialScopeName
                ),
                (
                    'gmd:thesaurusName gmd:CI_Citation',
                    ('gmd:title gmx:Anchor', {'xlink:href': lst + 'SpatialScope'}, 'Spatial scope'),
                    DATE('2019-05-22', 'publication')
                ),
            )

        if md.inspireTheme:
            yield (
                'gmd:descriptiveKeywords gmd:MD_Keywords',
                ('gmd:keyword gco:CharacterString', md.inspireThemeName),
                CODE('type', 'MD_KeywordTypeCode', 'theme'),
                (
                    'gmd:thesaurusName gmd:CI_Citation',
                    ('gmd:title gco:CharacterString', 'GEMET - INSPIRE themes, version 1.0'),
                    DATE('2008-06-01', 'publication')
                )
            )

        if md.keywords:
            yield (
                'gmd:descriptiveKeywords gmd:MD_Keywords',
                [('gmd:keyword gco:CharacterString', kw) for kw in md.keywords]
            )

        yield (
            'gmd:resourceConstraints gmd:MD_LegalConstraints',
            CODE('useConstraints', 'MD_RestrictionCode', 'otherRestrictions'),
            ('gmd:otherConstraints gco:CharacterString', md.accessConstraints),
            ('gmd:otherConstraints gco:CharacterString', md.license),
        )

        yield CODE('spatialRepresentationType', 'MD_SpatialRepresentationTypeCode', md.isoSpatialRepresentationType)

        yield (
            'gmd:spatialResolution gmd:MD_Resolution gmd:equivalentScale gmd:MD_RepresentativeFraction gmd:denominator gco:Integer',
            md.isoSpatialResolution
        )

        yield LANGUAGE()
        yield CODE('characterSet', 'MD_CharacterSetCode', 'utf8')

        if md.isoTopicCategory:
            yield 'gmd:topicCategory gmd:MD_TopicCategoryCode', md.isoTopicCategory

        if md.wgsExtent:
            yield 'gmd:extent gmd:EX_Extent gmd:geographicElement', BOUNDING_BOX(md.wgsExtent)

        if md.bounding_polygon_element:
            yield (
                'gmd:extent gmd:EX_Extent gmd:geographicElement gmd:EX_BoundingPolygon gmd:polygon',
                md.bounding_polygon_element
            )

        if md.dateBegin:
            yield (
                'gmd:extent gmd:EX_Extent gmd:temporalElement gmd:EX_TemporalExtent gmd:extent gml:TimePeriod',
                ('gml:beginPosition', md.dateBegin),
                ('gml:endPosition', md.dateEnd),
            )

    def distributionInfo():
        for link in md.extraLinks:
            if link.formatName:
                yield (
                    'gmd:distributionFormat gmd:MD_Format',
                    ('gmd:name gco:CharacterString', link.formatName),
                    ('gmd:version gco:CharacterString', link.formatVersion)
                )

        for link in md.extraLinks:
            yield (
                'gmd:transferOptions gmd:MD_DigitalTransferOptions',
                (
                    'gmd:onLine gmd:CI_OnlineResource',
                    ('gmd:linkage gmd:URL', ARGS.url_for(link.url)),
                    CODE('function', 'CI_OnLineFunctionCode', link.function)
                )
            )

    def dataQualityInfo():
        yield 'gmd:scope gmd:DQ_Scope', CODE('level', 'MD_ScopeCode', md.isoScope)

        if md.isoQualityConformanceQualityPass:
            yield (
                'gmd:report gmd:DQ_DomainConsistency gmd:result gmd:DQ_ConformanceResult',
                (
                    'gmd:specification gmd:CI_Citation',
                    ('gmd:title gco:CharacterString', md.isoQualityConformanceSpecificationTitle),
                    DATE(md.isoQualityConformanceSpecificationDate, 'publication')
                ),
                ('gmd:explanation gco:CharacterString', md.isoQualityConformanceExplanation),
                ('gmd:pass gco:Boolean', md.isoQualityConformanceQualityPass)
            )

        if md.isoQualityLineageStatement:
            yield (
                'gmd:lineage gmd:LI_Lineage',
                ('gmd:statement gco:CharacterString', md.isoQualityLineageStatement),
                (
                    'gmd:source gmd:LI_Source',
                    ('gmd:description gco:CharacterString', md.isoQualityLineageSource),
                    (
                        'gmd:scaleDenominator gmd:MD_RepresentativeFraction gmd:denominator gco:Integer',
                        md.isoQualityLineageSourceScale
                    )
                )
            )

    def content():
        yield 'gmd:fileIdentifier gco:CharacterString', md.catalogUid

        LANGUAGE()
        yield CODE('characterSet', 'MD_CharacterSetCode', 'utf8')

        yield CODE('hierarchyLevel', 'MD_ScopeCode', md.isoScope)
        yield 'gmd:hierarchyLevelName gco:CharacterString', md.isoScopeName

        yield 'gmd:contact', contact()

        yield 'gmd:dateStamp gco:Date', md.dateUpdated

        yield 'gmd:metadataStandardName gco:CharacterString', 'ISO19115'
        yield 'gmd:metadataStandardVersion gco:CharacterString', '2003/Cor.1:2006'

        if md.crs:
            yield (
                'gmd:referenceSystemInfo gmd:MD_ReferenceSystem gmd:referenceSystemIdentifier gmd:RS_Identifier gmd:code gco:CharacterString',
                md.crs.uri)

        yield 'gmd:identificationInfo gmd:MD_DataIdentification', identification()
        yield 'gmd:distributionInfo gmd:MD_Distribution', distributionInfo()
        yield 'gmd:dataQualityInfo gmd:DQ_DataQuality', dataQualityInfo()

    ##

    return 'gmd:MD_Metadata', content()
