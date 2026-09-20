""" "CSW Record template (gmd:MD_Metadata, ISO)."""

import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
from gws.lib.xmlx import tag

ML_GMX_CODELISTS = 'http://standards.iso.org/iso/19139/resources/gmxCodelists.xml'


def record(ta: server.TemplateArgs, md: gws.Metadata):
    def w_code(wrap, lst, value, text=None):
        return tag(
            f'gmd:{wrap}/gmd:{lst}',
            {'codeList': ML_GMX_CODELISTS + '#' + lst, 'codeListValue': value},
            text or value,
        )

    def w_date(d, typ):
        return tag(
            'gmd:date/gmd:CI_Date',
            tag('gmd:date/gco:Date', tpl.iso_date(d)),
            w_code(
                'dateType',
                'CI_DateTypeCode',
                typ,
            ),
        )

    def w_lang():
        return tag(
            'gmd:language/gmd:LanguageCode',
            {'codeList': 'http://www.loc.gov/standards/iso639-2/', 'codeListValue': md.language3},
            md.languageName,
        )

    def w_bbox(ext):
        return tag(
            'gmd:EX_GeographicBoundingBox',
            tag('gmd:westBoundLongitude/gco:Decimal', tpl.coord_dms(ext[0])),
            tag('gmd:eastBoundLongitude/gco:Decimal', tpl.coord_dms(ext[2])),
            tag('gmd:southBoundLatitude/gco:Decimal', tpl.coord_dms(ext[1])),
            tag('gmd:northBoundLatitude/gco:Decimal', tpl.coord_dms(ext[3])),
        )

    def contact():
        yield tag(
            'gmd:CI_ResponsibleParty',
            tag('gmd:organisationName/gco:CharacterString', md.contactOrganization),
            tag('gmd:positionName/gco:CharacterString', md.contactPosition),
            tag(
                'gmd:contactInfo/gmd:CI_Contact',
                tag(
                    'gmd:phone/gmd:CI_Telephone',
                    tag('gmd:voice/gco:CharacterString', md.contactPhone),
                    tag('gmd:facsimile/gco:CharacterString', md.contactFax),
                ),
                tag(
                    'gmd:address/gmd:CI_Address',
                    tag('gmd:deliveryPoint/gco:CharacterString', md.contactAddress),
                    tag('gmd:city/gco:CharacterString', md.contactCity),
                    tag('gmd:administrativeArea/gco:CharacterString', md.contactArea),
                    tag('gmd:postalCode/gco:CharacterString', md.contactZip),
                    tag('gmd:country/gco:CharacterString', md.contactCountry),
                    tag('gmd:electronicMailAddress/gco:CharacterString', md.contactEmail),
                ),
                tag('gmd:onlineResource/gmd:CI_OnlineResource/gmd:linkage/gmd:URL', md.contactUrl),
            ),
            w_code('role', 'CI_RoleCode', md.contactRole),
        )

    def identification():
        yield tag(
            'gmd:citation/gmd:CI_Citation',
            tag('gmd:title/gco:CharacterString', md.title),
            w_date(md.dateCreated, 'publication'),
            w_date(md.dateUpdated, 'revision'),
            tag('gmd:identifier/gmd:MD_Identifier/gmd:code/gco:CharacterString', md.catalogCitationUid),
        )

        yield tag('gmd:abstract/gco:CharacterString', md.abstract)

        yield tag('gmd:pointOfContact', contact())

        if md.inspireSpatialScope:
            lst = 'http://inspire.ec.europa.eu/metadata-codelist/SpatialScope/'
            yield tag(
                'gmd:descriptiveKeywords/gmd:MD_Keywords',
                tag('gmd:keyword/gmx:Anchor', {'xlink:href': lst + md.inspireSpatialScope}, md.inspireSpatialScopeName),
                tag(
                    'gmd:thesaurusName/gmd:CI_Citation',
                    tag('gmd:title/gmx:Anchor', {'xlink:href': lst + 'SpatialScope'}, 'Spatial scope'),
                    w_date('2019-05-22', 'publication'),
                ),
            )

        if md.inspireTheme:
            yield tag(
                'gmd:descriptiveKeywords/gmd:MD_Keywords',
                tag('gmd:keyword/gco:CharacterString', md.inspireThemeNameEn),
                w_code('type', 'MD_KeywordTypeCode', 'theme'),
                tag(
                    'gmd:thesaurusName/gmd:CI_Citation',
                    tag('gmd:title/gco:CharacterString', 'GEMET - INSPIRE themes, version 1.0'),
                    w_date('2008-06-01', 'publication'),
                ),
            )

        if md.keywords:
            yield tag(
                'gmd:descriptiveKeywords/gmd:MD_Keywords',
                [tag('gmd:keyword/gco:CharacterString', kw) for kw in md.keywords],
            )

        yield tag(
            'gmd:resourceConstraints/gmd:MD_LegalConstraints',
            w_code('useConstraints', 'MD_RestrictionCode', 'otherRestrictions'),
            tag('gmd:otherConstraints/gco:CharacterString', md.accessConstraints),
            tag('gmd:otherConstraints/gco:CharacterString', md.license),
        )

        yield w_code(
            'spatialRepresentationType',
            'MD_SpatialRepresentationTypeCode',
            md.isoSpatialRepresentationType,
        )

        if md.isoSpatialResolution:
            yield tag(
                'gmd:spatialResolution/gmd:MD_Resolution/gmd:equivalentScale/gmd:MD_RepresentativeFraction/gmd:denominator/gco:Integer',
                md.isoSpatialResolution,
            )

        yield w_lang()
        yield w_code('characterSet', 'MD_CharacterSetCode', 'utf8')

        if md.isoTopicCategories:
            for cat in md.isoTopicCategories:
                yield tag('gmd:topicCategory/gmd:MD_TopicCategoryCode', cat)

        if md.wgsExtent:
            yield tag('gmd:extent/gmd:EX_Extent/gmd:geographicElement', w_bbox(md.wgsExtent))

        # @TODO
        # if md.bounding_polygon_element:
        #     yield (
        #         'gmd:extent/gmd:EX_Extent/gmd:geographicElement/gmd:EX_BoundingPolygon/gmd:polygon',
        #         md.bounding_polygon_element
        #     )

        if md.temporalBegin:
            yield tag(
                'gmd:extent/gmd:EX_Extent/gmd:temporalElement/gmd:EX_TemporalExtent/gmd:extent/gml:TimePeriod',
                tag('gml:beginPosition', md.temporalBegin),
                tag('gml:endPosition', md.temporalEnd),
            )

    def distributionInfo():
        for link in md.metaLinks:
            if link.format:
                yield tag(
                    'gmd:distributionFormat/gmd:MD_Format',
                    tag('gmd:name/gco:CharacterString', link.format),
                    tag('gmd:version/gco:CharacterString', link.formatVersion),
                )

        for link in md.metaLinks:
            yield tag(
                'gmd:transferOptions/gmd:MD_DigitalTransferOptions',
                tag(
                    'gmd:onLine/gmd:CI_OnlineResource',
                    tag('gmd:linkage/gmd:URL', ta.url_for(link.url)),
                    w_code('function', 'CI_OnLineFunctionCode', link.function),
                ),
            )

    def dataQualityInfo():
        yield tag('gmd:scope/gmd:DQ_Scope', w_code('level', 'MD_ScopeCode', md.isoScope))

        if md.isoQualityConformanceQualityPass:
            yield tag(
                'gmd:report/gmd:DQ_DomainConsistency/gmd:result/gmd:DQ_ConformanceResult',
                tag(
                    'gmd:specification/gmd:CI_Citation',
                    tag('gmd:title/gco:CharacterString', md.isoQualityConformanceSpecificationTitle),
                    w_date(md.isoQualityConformanceSpecificationDate, 'publication'),
                ),
                tag('gmd:explanation/gco:CharacterString', md.isoQualityConformanceExplanation),
                tag('gmd:pass/gco:Boolean', md.isoQualityConformanceQualityPass),
            )

        if md.isoQualityLineageStatement:
            yield tag(
                'gmd:lineage/gmd:LI_Lineage',
                tag('gmd:statement/gco:CharacterString', md.isoQualityLineageStatement),
                tag(
                    'gmd:source/gmd:LI_Source',
                    tag('gmd:description/gco:CharacterString', md.isoQualityLineageSource),
                    tag('gmd:scaleDenominator/gmd:MD_RepresentativeFraction/gmd:denominator/gco:Integer', md.isoQualityLineageSourceScale),
                ),
            )

    def content():
        yield tag('gmd:fileIdentifier/gco:CharacterString', md.catalogUid)

        w_lang()
        yield w_code('characterSet', 'MD_CharacterSetCode', 'utf8')

        yield w_code('hierarchyLevel', 'MD_ScopeCode', md.isoScope)
        yield tag('gmd:hierarchyLevelName/gco:CharacterString', md.isoScopeName)

        yield tag('gmd:contact', contact())

        yield tag('gmd:dateStamp/gco:Date', tpl.iso_date(md.dateUpdated))

        yield tag('gmd:metadataStandardName/gco:CharacterString', 'ISO19115')
        yield tag('gmd:metadataStandardVersion/gco:CharacterString', '2003/Cor.1:2006')

        if md.crs:
            yield tag(
                'gmd:referenceSystemInfo/gmd:MD_ReferenceSystem/gmd:referenceSystemIdentifier/gmd:RS_Identifier/gmd:code/gco:CharacterString',
                md.crs.uri,
            )

        yield tag('gmd:identificationInfo/gmd:MD_DataIdentification', identification())
        yield tag('gmd:distributionInfo/gmd:MD_Distribution', distributionInfo())
        yield tag('gmd:dataQualityInfo/gmd:DQ_DataQuality', dataQualityInfo())

    ##

    return tag('gmd:MD_Metadata', content())
