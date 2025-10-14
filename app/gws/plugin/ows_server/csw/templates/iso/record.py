""" "CSW Record template (gmd:MD_Metadata, ISO)."""

import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl

ML_GMX_CODELISTS = 'http://standards.iso.org/iso/19139/resources/gmxCodelists.xml'


def record(ta: server.TemplateArgs, md: gws.Metadata):
    def w_code(wrap, lst, value, text=None):
        return (
            f'gmd:{wrap}/gmd:{lst}',
            {'codeList': ML_GMX_CODELISTS + '#' + lst, 'codeListValue': value},
            text or value,
        )

    def w_date(d, typ):
        return (
            'gmd:date/gmd:CI_Date',
            ('gmd:date/gco:Date', tpl.iso_date(d)),
            w_code(
                'dateType',
                'CI_DateTypeCode',
                typ,
            ),
        )

    def w_lang():
        return (
            'gmd:language/gmd:LanguageCode',
            {'codeList': 'http://www.loc.gov/standards/iso639-2/', 'codeListValue': md.language3},
            md.languageName,
        )

    def w_bbox(ext):
        return (
            'gmd:EX_GeographicBoundingBox',
            ('gmd:westBoundLongitude/gco:Decimal', tpl.coord_dms(ext[0])),
            ('gmd:eastBoundLongitude/gco:Decimal', tpl.coord_dms(ext[2])),
            ('gmd:southBoundLatitude/gco:Decimal', tpl.coord_dms(ext[1])),
            ('gmd:northBoundLatitude/gco:Decimal', tpl.coord_dms(ext[3])),
        )

    def contact():
        yield (
            'gmd:CI_ResponsibleParty',
            ('gmd:organisationName/gco:CharacterString', md.contactOrganization),
            ('gmd:positionName/gco:CharacterString', md.contactPosition),
            (
                'gmd:contactInfo/gmd:CI_Contact',
                (
                    'gmd:phone/gmd:CI_Telephone',
                    ('gmd:voice/gco:CharacterString', md.contactPhone),
                    ('gmd:facsimile/gco:CharacterString', md.contactFax),
                ),
                (
                    'gmd:address/gmd:CI_Address',
                    ('gmd:deliveryPoint/gco:CharacterString', md.contactAddress),
                    ('gmd:city/gco:CharacterString', md.contactCity),
                    ('gmd:administrativeArea/gco:CharacterString', md.contactArea),
                    ('gmd:postalCode/gco:CharacterString', md.contactZip),
                    ('gmd:country/gco:CharacterString', md.contactCountry),
                    ('gmd:electronicMailAddress/gco:CharacterString', md.contactEmail),
                ),
                ('gmd:onlineResource/gmd:CI_OnlineResource/gmd:linkage/gmd:URL', md.contactUrl),
            ),
            w_code('role', 'CI_RoleCode', md.contactRole),
        )

    def identification():
        yield (
            'gmd:citation/gmd:CI_Citation',
            ('gmd:title/gco:CharacterString', md.title),
            w_date(md.dateCreated, 'publication'),
            w_date(md.dateUpdated, 'revision'),
            ('gmd:identifier/gmd:MD_Identifier/gmd:code/gco:CharacterString', md.catalogCitationUid),
        )

        yield 'gmd:abstract/gco:CharacterString', md.abstract

        yield 'gmd:pointOfContact', contact()

        if md.inspireSpatialScope:
            lst = 'http://inspire.ec.europa.eu/metadata-codelist/SpatialScope/'
            yield (
                'gmd:descriptiveKeywords/gmd:MD_Keywords',
                ('gmd:keyword/gmx:Anchor', {'xlink:href': lst + md.inspireSpatialScope}, md.inspireSpatialScopeName),
                (
                    'gmd:thesaurusName/gmd:CI_Citation',
                    ('gmd:title/gmx:Anchor', {'xlink:href': lst + 'SpatialScope'}, 'Spatial scope'),
                    w_date('2019-05-22', 'publication'),
                ),
            )

        if md.inspireTheme:
            yield (
                'gmd:descriptiveKeywords/gmd:MD_Keywords',
                ('gmd:keyword/gco:CharacterString', md.inspireThemeNameEn),
                w_code('type', 'MD_KeywordTypeCode', 'theme'),
                (
                    'gmd:thesaurusName/gmd:CI_Citation',
                    ('gmd:title/gco:CharacterString', 'GEMET - INSPIRE themes, version 1.0'),
                    w_date('2008-06-01', 'publication'),
                ),
            )

        if md.keywords:
            yield (
                'gmd:descriptiveKeywords/gmd:MD_Keywords',
                [('gmd:keyword/gco:CharacterString', kw) for kw in md.keywords],
            )

        yield (
            'gmd:resourceConstraints/gmd:MD_LegalConstraints',
            w_code('useConstraints', 'MD_RestrictionCode', 'otherRestrictions'),
            ('gmd:otherConstraints/gco:CharacterString', md.accessConstraints),
            ('gmd:otherConstraints/gco:CharacterString', md.license),
        )

        yield w_code(
            'spatialRepresentationType',
            'MD_SpatialRepresentationTypeCode',
            md.isoSpatialRepresentationType,
        )

        if md.isoSpatialResolution:
            yield (
                'gmd:spatialResolution/gmd:MD_Resolution/gmd:equivalentScale/gmd:MD_RepresentativeFraction/gmd:denominator/gco:Integer',
                md.isoSpatialResolution,
            )

        yield w_lang()
        yield w_code('characterSet', 'MD_CharacterSetCode', 'utf8')

        if md.isoTopicCategories:
            for cat in md.isoTopicCategories:
                yield 'gmd:topicCategory/gmd:MD_TopicCategoryCode', cat

        if md.wgsExtent:
            yield 'gmd:extent/gmd:EX_Extent/gmd:geographicElement', w_bbox(md.wgsExtent)

        # @TODO
        # if md.bounding_polygon_element:
        #     yield (
        #         'gmd:extent/gmd:EX_Extent/gmd:geographicElement/gmd:EX_BoundingPolygon/gmd:polygon',
        #         md.bounding_polygon_element
        #     )

        if md.temporalBegin:
            yield (
                'gmd:extent/gmd:EX_Extent/gmd:temporalElement/gmd:EX_TemporalExtent/gmd:extent/gml:TimePeriod',
                ('gml:beginPosition', md.temporalBegin),
                ('gml:endPosition', md.temporalEnd),
            )

    def distributionInfo():
        for link in md.metaLinks:
            if link.format:
                yield (
                    'gmd:distributionFormat/gmd:MD_Format',
                    ('gmd:name/gco:CharacterString', link.format),
                    ('gmd:version/gco:CharacterString', link.formatVersion),
                )

        for link in md.metaLinks:
            yield (
                'gmd:transferOptions/gmd:MD_DigitalTransferOptions',
                (
                    'gmd:onLine/gmd:CI_OnlineResource',
                    ('gmd:linkage/gmd:URL', ta.url_for(link.url)),
                    w_code('function', 'CI_OnLineFunctionCode', link.function),
                ),
            )

    def dataQualityInfo():
        yield 'gmd:scope/gmd:DQ_Scope', w_code('level', 'MD_ScopeCode', md.isoScope)

        if md.isoQualityConformanceQualityPass:
            yield (
                'gmd:report/gmd:DQ_DomainConsistency/gmd:result/gmd:DQ_ConformanceResult',
                (
                    'gmd:specification/gmd:CI_Citation',
                    ('gmd:title/gco:CharacterString', md.isoQualityConformanceSpecificationTitle),
                    w_date(md.isoQualityConformanceSpecificationDate, 'publication'),
                ),
                ('gmd:explanation/gco:CharacterString', md.isoQualityConformanceExplanation),
                ('gmd:pass/gco:Boolean', md.isoQualityConformanceQualityPass),
            )

        if md.isoQualityLineageStatement:
            yield (
                'gmd:lineage/gmd:LI_Lineage',
                ('gmd:statement/gco:CharacterString', md.isoQualityLineageStatement),
                (
                    'gmd:source/gmd:LI_Source',
                    ('gmd:description/gco:CharacterString', md.isoQualityLineageSource),
                    ('gmd:scaleDenominator/gmd:MD_RepresentativeFraction/gmd:denominator/gco:Integer', md.isoQualityLineageSourceScale),
                ),
            )

    def content():
        yield 'gmd:fileIdentifier/gco:CharacterString', md.catalogUid

        w_lang()
        yield w_code('characterSet', 'MD_CharacterSetCode', 'utf8')

        yield w_code('hierarchyLevel', 'MD_ScopeCode', md.isoScope)
        yield 'gmd:hierarchyLevelName/gco:CharacterString', md.isoScopeName

        yield 'gmd:contact', contact()

        yield 'gmd:dateStamp/gco:Date', tpl.iso_date(md.dateUpdated)

        yield 'gmd:metadataStandardName/gco:CharacterString', 'ISO19115'
        yield 'gmd:metadataStandardVersion/gco:CharacterString', '2003/Cor.1:2006'

        if md.crs:
            yield (
                'gmd:referenceSystemInfo/gmd:MD_ReferenceSystem/gmd:referenceSystemIdentifier/gmd:RS_Identifier/gmd:code/gco:CharacterString',
                md.crs.uri,
            )

        yield 'gmd:identificationInfo/gmd:MD_DataIdentification', identification()
        yield 'gmd:distributionInfo/gmd:MD_Distribution', distributionInfo()
        yield 'gmd:dataQualityInfo/gmd:DQ_DataQuality', dataQualityInfo()

    ##

    return 'gmd:MD_Metadata', content()
