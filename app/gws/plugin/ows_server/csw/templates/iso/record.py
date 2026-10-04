"""CSW record builder for the ISO profile (``gmd:MD_Metadata``)."""

import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
from gws.lib.xmlx import tag

ML_GMX_CODELISTS = 'http://standards.iso.org/iso/19139/resources/gmxCodelists.xml'


def record(ta: server.TemplateArgs, md: gws.Metadata):
    """Create a ``gmd:MD_Metadata`` element for a metadata object.

    Args:
        ta: Template arguments.
        md: Metadata of the catalog record.

    Returns:
        The XML element.
    """
    def w_code(wrap, lst, value, text=None):
        return tag(
            f'GMD:{wrap}/GMD:{lst}',
            {'codeList': ML_GMX_CODELISTS + '#' + lst, 'codeListValue': value},
            text or value,
        )

    def w_date(d, typ):
        return tag(
            'GMD:date/GMD:CI_Date',
            tag('GMD:date/GCO:Date', tpl.iso_date(d)),
            w_code(
                'dateType',
                'CI_DateTypeCode',
                typ,
            ),
        )

    def w_lang():
        return tag(
            'GMD:language/GMD:LanguageCode',
            {'codeList': 'http://www.loc.gov/standards/iso639-2/', 'codeListValue': md.language3},
            md.languageName,
        )

    def w_bbox(ext):
        return tag(
            'GMD:EX_GeographicBoundingBox',
            tag('GMD:westBoundLongitude/GCO:Decimal', tpl.coord_dms(ext[0])),
            tag('GMD:eastBoundLongitude/GCO:Decimal', tpl.coord_dms(ext[2])),
            tag('GMD:southBoundLatitude/GCO:Decimal', tpl.coord_dms(ext[1])),
            tag('GMD:northBoundLatitude/GCO:Decimal', tpl.coord_dms(ext[3])),
        )

    def contact():
        yield tag(
            'GMD:CI_ResponsibleParty',
            tag('GMD:organisationName/GCO:CharacterString', md.contactOrganization),
            tag('GMD:positionName/GCO:CharacterString', md.contactPosition),
            tag(
                'GMD:contactInfo/GMD:CI_Contact',
                tag(
                    'GMD:phone/GMD:CI_Telephone',
                    tag('GMD:voice/GCO:CharacterString', md.contactPhone),
                    tag('GMD:facsimile/GCO:CharacterString', md.contactFax),
                ),
                tag(
                    'GMD:address/GMD:CI_Address',
                    tag('GMD:deliveryPoint/GCO:CharacterString', md.contactAddress),
                    tag('GMD:city/GCO:CharacterString', md.contactCity),
                    tag('GMD:administrativeArea/GCO:CharacterString', md.contactArea),
                    tag('GMD:postalCode/GCO:CharacterString', md.contactZip),
                    tag('GMD:country/GCO:CharacterString', md.contactCountry),
                    tag('GMD:electronicMailAddress/GCO:CharacterString', md.contactEmail),
                ),
                tag('GMD:onlineResource/GMD:CI_OnlineResource/GMD:linkage/GMD:URL', md.contactUrl),
            ),
            w_code('role', 'CI_RoleCode', md.contactRole),
        )

    def identification():
        yield tag(
            'GMD:citation/GMD:CI_Citation',
            tag('GMD:title/GCO:CharacterString', md.title),
            w_date(md.dateCreated, 'publication'),
            w_date(md.dateUpdated, 'revision'),
            tag('GMD:identifier/GMD:MD_Identifier/GMD:code/GCO:CharacterString', md.catalogCitationUid),
        )

        yield tag('GMD:abstract/GCO:CharacterString', md.abstract)

        yield tag('GMD:pointOfContact', contact())

        if md.inspireSpatialScope:
            lst = 'http://inspire.ec.europa.eu/metadata-codelist/SpatialScope/'
            yield tag(
                'GMD:descriptiveKeywords/GMD:MD_Keywords',
                tag('GMD:keyword/GMX:Anchor', {'XLINK:href': lst + md.inspireSpatialScope}, md.inspireSpatialScopeName),
                tag(
                    'GMD:thesaurusName/GMD:CI_Citation',
                    tag('GMD:title/GMX:Anchor', {'XLINK:href': lst + 'SpatialScope'}, 'Spatial scope'),
                    w_date('2019-05-22', 'publication'),
                ),
            )

        if md.inspireTheme:
            yield tag(
                'GMD:descriptiveKeywords/GMD:MD_Keywords',
                tag('GMD:keyword/GCO:CharacterString', md.inspireThemeNameEn),
                w_code('type', 'MD_KeywordTypeCode', 'theme'),
                tag(
                    'GMD:thesaurusName/GMD:CI_Citation',
                    tag('GMD:title/GCO:CharacterString', 'GEMET - INSPIRE themes, version 1.0'),
                    w_date('2008-06-01', 'publication'),
                ),
            )

        if md.keywords:
            yield tag(
                'GMD:descriptiveKeywords/GMD:MD_Keywords',
                [tag('GMD:keyword/GCO:CharacterString', kw) for kw in md.keywords],
            )

        yield tag(
            'GMD:resourceConstraints/GMD:MD_LegalConstraints',
            w_code('useConstraints', 'MD_RestrictionCode', 'otherRestrictions'),
            tag('GMD:otherConstraints/GCO:CharacterString', md.accessConstraints),
            tag('GMD:otherConstraints/GCO:CharacterString', md.license),
        )

        yield w_code(
            'spatialRepresentationType',
            'MD_SpatialRepresentationTypeCode',
            md.isoSpatialRepresentationType,
        )

        if md.isoSpatialResolution:
            yield tag(
                'GMD:spatialResolution/GMD:MD_Resolution/GMD:equivalentScale/GMD:MD_RepresentativeFraction/GMD:denominator/GCO:Integer',
                md.isoSpatialResolution,
            )

        yield w_lang()
        yield w_code('characterSet', 'MD_CharacterSetCode', 'utf8')

        if md.isoTopicCategories:
            for cat in md.isoTopicCategories:
                yield tag('GMD:topicCategory/GMD:MD_TopicCategoryCode', cat)

        if md.wgsExtent:
            yield tag('GMD:extent/GMD:EX_Extent/GMD:geographicElement', w_bbox(md.wgsExtent))

        # @TODO
        # if md.bounding_polygon_element:
        #     yield (
        #         'GMD:extent/GMD:EX_Extent/GMD:geographicElement/GMD:EX_BoundingPolygon/GMD:polygon',
        #         md.bounding_polygon_element
        #     )

        if md.temporalBegin:
            yield tag(
                'GMD:extent/GMD:EX_Extent/GMD:temporalElement/GMD:EX_TemporalExtent/GMD:extent/GML:TimePeriod',
                tag('GML:beginPosition', md.temporalBegin),
                tag('GML:endPosition', md.temporalEnd),
            )

    def distributionInfo():
        for link in md.metaLinks:
            if link.format:
                yield tag(
                    'GMD:distributionFormat/GMD:MD_Format',
                    tag('GMD:name/GCO:CharacterString', link.format),
                    tag('GMD:version/GCO:CharacterString', link.formatVersion),
                )

        for link in md.metaLinks:
            yield tag(
                'GMD:transferOptions/GMD:MD_DigitalTransferOptions',
                tag(
                    'GMD:onLine/GMD:CI_OnlineResource',
                    tag('GMD:linkage/GMD:URL', ta.url_for(link.url)),
                    w_code('function', 'CI_OnLineFunctionCode', link.function),
                ),
            )

    def dataQualityInfo():
        yield tag('GMD:scope/GMD:DQ_Scope', w_code('level', 'MD_ScopeCode', md.isoScope))

        if md.isoQualityConformanceQualityPass:
            yield tag(
                'GMD:report/GMD:DQ_DomainConsistency/GMD:result/GMD:DQ_ConformanceResult',
                tag(
                    'GMD:specification/GMD:CI_Citation',
                    tag('GMD:title/GCO:CharacterString', md.isoQualityConformanceSpecificationTitle),
                    w_date(md.isoQualityConformanceSpecificationDate, 'publication'),
                ),
                tag('GMD:explanation/GCO:CharacterString', md.isoQualityConformanceExplanation),
                tag('GMD:pass/GCO:Boolean', md.isoQualityConformanceQualityPass),
            )

        if md.isoQualityLineageStatement:
            yield tag(
                'GMD:lineage/GMD:LI_Lineage',
                tag('GMD:statement/GCO:CharacterString', md.isoQualityLineageStatement),
                tag(
                    'GMD:source/GMD:LI_Source',
                    tag('GMD:description/GCO:CharacterString', md.isoQualityLineageSource),
                    tag('GMD:scaleDenominator/GMD:MD_RepresentativeFraction/GMD:denominator/GCO:Integer', md.isoQualityLineageSourceScale),
                ),
            )

    def content():
        yield tag('GMD:fileIdentifier/GCO:CharacterString', md.catalogUid)

        w_lang()
        yield w_code('characterSet', 'MD_CharacterSetCode', 'utf8')

        yield w_code('hierarchyLevel', 'MD_ScopeCode', md.isoScope)
        yield tag('GMD:hierarchyLevelName/GCO:CharacterString', md.isoScopeName)

        yield tag('GMD:contact', contact())

        yield tag('GMD:dateStamp/GCO:Date', tpl.iso_date(md.dateUpdated))

        yield tag('GMD:metadataStandardName/GCO:CharacterString', 'ISO19115')
        yield tag('GMD:metadataStandardVersion/GCO:CharacterString', '2003/Cor.1:2006')

        if md.crs:
            yield tag(
                'GMD:referenceSystemInfo/GMD:MD_ReferenceSystem/GMD:referenceSystemIdentifier/GMD:RS_Identifier/GMD:code/GCO:CharacterString',
                md.crs.uri,
            )

        yield tag('GMD:identificationInfo/GMD:MD_DataIdentification', identification())
        yield tag('GMD:distributionInfo/GMD:MD_Distribution', distributionInfo())
        yield tag('GMD:dataQualityInfo/GMD:DQ_DataQuality', dataQualityInfo())

    ##

    return tag('GMD:MD_Metadata', content())
