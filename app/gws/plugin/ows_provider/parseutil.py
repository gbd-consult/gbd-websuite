"""Parse utilities for OWS XML files."""

import re

import gws
import gws.gis.crs
import gws.gis.extent
import gws.lib.metadata
import gws.lib.xml2 as xml2
import gws.types as t


def service_operations(root_el: gws.XmlElement) -> t.List[gws.OwsOperation]:
    # <ows:OperationsMetadata>
    #     <ows:Operation name="GetCapabilities">
    #         <ows:DCP><ows:HTTP><ows:Get xlink:href="...."/></ows:HTTP></ows:DCP>
    #         <ows:Parameter name="AcceptVersions">
    #             <ows:AllowedValues><ows:Value>2.0.0</ows:Value></ows:AllowedValues>

    els = xml2.all(root_el, 'OperationsMetadata Operation')
    if els:
        return [_parse_operation(e) for e in els]

    # <Capability>
    #   <Request>
    #     <GetMap>
    #       <Format>image/png</Format>
    #       <Format>application/atom xml</Format>
    #       <DCPType><HTTP><Get><OnlineResource ....

    el = xml2.first(root_el, 'Capability Request')
    if el:
        return [_parse_operation(e) for e in el.children]

    return []


def _parse_operation(el: gws.XmlElement) -> gws.OwsOperation:
    params = {}

    for param_el in xml2.all(el, 'Parameter'):
        values = xml2.text_list(param_el, 'Value', 'AllowedValues Value')
        if values:
            params[xml2.attr(param_el, 'name')] = values

    op = gws.OwsOperation(
        verb=xml2.attr(el, 'name') or el.name,
        formats=xml2.text_list(el, 'Format'),
        get_url=_parse_url(xml2.first(el, 'DCP HTTP Get', 'DCPType HTTP Get')),
        post_url=_parse_url(xml2.first(el, 'DCP HTTP Post', 'DCPType HTTP Post')),
        params=params,
    )

    if 'outputFormat' in params:
        op.formats.extend(params['outputFormat'])

    return op


##


def service_metadata(root_el: gws.XmlElement) -> gws.lib.metadata.Metadata:
    # wms
    #
    #   <Capabilities
    #       <Service...
    #           <Name>...
    #           <Title>...
    #           <ContactInformation>...
    #
    # ows
    #
    #   <Capabilities
    #       <ows:ServiceIdentification>
    #           <ows:Title>....
    #       <ows:ServiceProvider>
    #           <ows:ProviderName>...
    #           <ows:ServiceContact>...

    d = _metadata_dict(xml2.first(root_el, 'Service', 'ServiceIdentification'))
    d.update(_contact_dict(root_el))
    d['contactProviderName'] = xml2.text(root_el, 'ServiceProvider ProviderName')
    d['contactProviderSite'] = xml2.text(root_el, 'ServiceProvider ProviderSite')

    #   <Capabilities
    #       <ServiceMetadataURL

    link = _parse_link(xml2.first(root_el, 'ServiceMetadataURL'))
    if link:
        d['metaLinks'] = [link]

    return gws.lib.metadata.from_dict(gws.strip(d))


def element_metadata(el: gws.XmlElement) -> gws.lib.metadata.Metadata:
    #   <whatever, e.g. Layer or FeatureType
    #       <Name...
    #       <Title...

    return gws.lib.metadata.from_dict(_metadata_dict(el))


_contact_mapping = [
    # wms

    ('contactArea', 'StateOrProvince'),
    ('contactCity', 'City'),
    ('contactCountry', 'Country'),
    ('contactEmail', 'ContactElectronicMailAddress'),
    ('contactFax', 'ContactFacsimileTelephone'),
    ('contactOrganization', 'ContactOrganization'),
    ('contactPerson', 'ContactPerson'),
    ('contactPhone', 'ContactVoiceTelephone'),
    ('contactPosition', 'ContactPosition'),
    ('contactZip', 'PostCode'),

    # ows

    ('contactArea', 'AdministrativeArea'),
    ('contactCity', 'City'),
    ('contactCountry', 'Country'),
    ('contactEmail', 'ElectronicMailAddress'),
    ('contactFax', 'Facsimile'),
    ('contactOrganization', 'ProviderName'),
    ('contactPerson', 'IndividualName'),
    ('contactPhone', 'Voice'),
    ('contactPosition', 'PositionName'),
    ('contactZip', 'PostalCode'),
]


def _contact_dict(el: gws.XmlElement) -> dict:
    contact_el = xml2.first(el, 'Service ContactInformation', 'ServiceProvider ServiceContact')
    if not contact_el:
        return {}

    texts = xml2.text_dict(contact_el, deep=True)
    d = {}

    for dst, src in _contact_mapping:
        val = texts.get(src, '').strip()
        if val:
            d[dst] = val

    return d


def _metadata_dict(el: gws.XmlElement) -> dict:
    if not el:
        return {}

    d = {
        'abstract': xml2.text(el, 'Abstract'),
        'accessConstraints': xml2.text(el, 'AccessConstraints'),
        'attribution': xml2.text(el, 'Attribution Title'),
        'fees': xml2.text(el, 'Fees'),
        'keywords': xml2.text_list(el, 'Keywords', 'KeywordList', deep=True),
        'name': xml2.text(el, 'Name') or xml2.text(el, 'Identifier'),
        'title': xml2.text(el, 'Title'),
        'metaLinks': gws.compact(_parse_link(e) for e in xml2.all(el, 'MetadataURL')),
    }

    e = xml2.first(el, 'AuthorityURL')
    if e:
        d['authorityUrl'] = _parse_url(e)
        d['authorityName'] = xml2.attr(e, 'name')

    e = xml2.first(el, 'Identifier')
    if e:
        d['authorityIdentifier'] = e.text

    return gws.strip(d)


##

def supported_bounds(layer_el: gws.XmlElement, extra_crsids: t.Optional[t.List[str]] = None) -> t.List[gws.Bounds]:
    # <Layer...
    #     <CRS>EPSG....
    #     <EX_GeographicBoundingBox...
    #     <BoundingBox CRS="EPSG:" minx=....
    # 
    # <FeatureType...
    #     <DefaultCRS>urn:ogc:def:crs:EPSG...
    #     <OtherCRS>urn:ogc:def:crs:EPSG...
    #     <ows:WGS84BoundingBox>
    #         <ows:LowerCorner...
    #         <ows:UpperCorner...

    if not layer_el:
        return []

    crs_to_bounds = {}

    # enumerate explicitly listed bounds (WMS)

    for e in xml2.all(layer_el, 'BoundingBox'):
        crs = gws.gis.crs.get(xml2.attr(e, 'srs') or xml2.attr(e, 'crs'))
        bbox = _parse_bbox(e)
        if crs and bbox:
            crs_to_bounds[crs] = gws.Bounds(crs=crs, extent=bbox)

    # NB prefer these for 4326 to avoid axis issues

    e = xml2.first(layer_el, 'EX_GeographicBoundingBox', 'WGS84BoundingBox', 'LatLonBoundingBox')
    if e:
        bbox = _parse_bbox(e)
        if bbox:
            wgs = gws.gis.crs.get4326()
            crs_to_bounds[wgs] = gws.Bounds(crs=wgs, extent=bbox)

    # no bounds

    if not crs_to_bounds:
        return []

    # collect other supported crs and add extras (e.g. wmts matrix sets)

    crsids = set()

    for tag in 'DefaultSRS', 'DefaultCRS', 'OtherSRS', 'OtherCRS', 'SRS', 'CRS':
        for e in xml2.all(layer_el, tag):
            if e.text:
                crsids.add(e.text)

    if extra_crsids:
        crsids.update(extra_crsids)

    # freeze the bounds list to prevent double reprojection

    bs = list(crs_to_bounds.values())

    # compute bounds for those without bounds

    for crsid in crsids:
        new_crs = gws.gis.crs.get(crsid)
        if not new_crs or new_crs in crs_to_bounds:
            continue
        bb = gws.gis.crs.best_bounds(new_crs, bs)
        try:
            new_ext = gws.gis.extent.transform(bb.extent, bb.crs, new_crs)
        except Exception as exc:
            gws.log.error(f'failed transform {bb.crs.srid}=>{new_crs.srid}')
            continue
        crs_to_bounds[new_crs] = gws.Bounds(crs=new_crs, extent=new_ext)

    return list(crs_to_bounds.values())


##


def parse_style(el: gws.XmlElement) -> gws.SourceStyle:
    # <Style>
    #     <Name>default...
    #     <Title>...
    #     <LegendURL
    #         <Format>...
    #         <OnlineResource...

    st = gws.SourceStyle()

    st.metadata = element_metadata(el)
    st.name = st.metadata.get('name', '').lower()
    st.legend_url = _parse_url(xml2.first(el, 'LegendURL'))
    st.is_default = (
            xml2.attr(el, 'IsDefault') == 'true'
            or st.name == 'default'
            or st.name.endswith(':default'))
    return st


def default_style(styles: t.List[gws.SourceStyle]) -> t.Optional[gws.SourceStyle]:
    for s in styles:
        if s.is_default:
            return s
    return styles[0] if styles else None


##


def to_float(s, default=0.0):
    return float(s or default)


def to_int(s, default=0):
    # accept floats as well, but convert to int
    return int(float(s or default))


def to_float_pair(s):
    s = s.split()
    return float(s[0]), float(s[1])


##


def _parse_bbox(el: gws.XmlElement):
    # note: bboxes are always converted to (x1, y1, x2, y2) with x1 < x2, y1 < y2

    # <BoundingBox/LatLonBoundingBox CRS="..." minx="0" miny="1" maxx="2" maxy="3"/>

    if xml2.attr(el, 'minx'):
        return [
            to_float(xml2.attr(el, 'minx')),
            to_float(xml2.attr(el, 'miny')),
            to_float(xml2.attr(el, 'maxx')),
            to_float(xml2.attr(el, 'maxy')),
        ]

    # <ows:BoundingBox/WGS84BoundingBox
    #       <ows:LowerCorner> 0 1
    #       <ows:UpperCorner> 2 3

    if xml2.first(el, 'LowerCorner'):
        x1, y1 = to_float_pair(xml2.text(el, 'LowerCorner'))
        x2, y2 = to_float_pair(xml2.text(el, 'UpperCorner'))
        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]

    # <EX_GeographicBoundingBox>
    #       <westBoundLongitude> 0
    #       <eastBoundLongitude> 2
    #       <southBoundLatitude> 1
    #       <northBoundLatitude> 3

    if xml2.first(el, 'westBoundLongitude'):
        x1 = to_float(xml2.text(el, 'eastBoundLongitude'))
        y1 = to_float(xml2.text(el, 'southBoundLatitude'))
        x2 = to_float(xml2.text(el, 'westBoundLongitude'))
        y2 = to_float(xml2.text(el, 'northBoundLatitude'))
        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]


def _parse_url(el: gws.XmlElement) -> str:
    def cleanup(s):
        return (s or '').strip(' ?&')

    # <ows:DCP>
    #       <ows:HTTP>
    #           <ows:Get xlink:href=... <-- we are here

    s = xml2.attr(el, 'href') or xml2.attr(el, 'onlineResource')
    if s:
        return cleanup(s)

    # <whatever <--
    #       <OnlineResource xlink:href=...

    e = xml2.first(el, 'OnlineResource')
    if e:
        return cleanup(xml2.attr(e, 'href', default=e.text))

    return ''


def _parse_link(el: gws.XmlElement) -> t.Optional[gws.MetadataLink]:
    # <MetadataURL type="...
    #       <Format...
    # 	    <OnlineResource...

    if not el:
        return None

    d = gws.strip({
        'url': _parse_url(el),
        'type': xml2.attr(el, 'type'),
        'formatName': xml2.text(el, 'Format'),
    })

    if d:
        return gws.MetadataLink(d)
