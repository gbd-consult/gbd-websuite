"""Parse utilities for OWS XML files."""

import re

import gws
import gws.gis.crs
import gws.gis.extent
import gws.lib.xmlx as xmlx
import gws.types as t


def service_operations(caps_el: gws.IXmlElement) -> t.List[gws.OwsOperation]:
    # <ows:OperationsMetadata>
    #     <ows:Operation name="GetCapabilities">...

    els = caps_el.findall('OperationsMetadata/Operation')
    if els:
        return [_parse_operation(e) for e in els]

    # <Capability>
    #   <Request>
    #     <GetCapabilities>...

    el = caps_el.find('Capability/Request')
    if el:
        return [_parse_operation(e) for e in el]

    return []


def _parse_operation(el: gws.IXmlElement) -> gws.OwsOperation:
    params = {}

    # <Parameter name="Format">
    #   <AllowedValues>
    #       <Value>image/gif</Value>
    #       <Value>image/png</Value>
    #       <Value>image/jpeg</Value>
    #   </AllowedValues>
    # </Parameter>

    for param_el in el.findall('Parameter'):
        values = param_el.text_list('Value') + param_el.text_list('AllowedValues/Value')
        if values:
            params[param_el.get('name')] = values

    # <Operation name="GetMap">
    #     <DCP> <HTTP>
    #         <Get xlink:href="...."/>
    #         <Post xlink:href="..."/>
    #     </HTTP> </DCP>
    #
    #
    # <GetMap>
    #     <Format>image/png</Format>
    #     <DCPType> <HTTP> <Get>
    #         <OnlineResource xlink:type="simple" xlink:href="..."/>
    #     </Get> </HTTP> </DCPType>
    # </GetMap>

    op = gws.OwsOperation(
        verb=el.get('name') or el.tag,
        formats=el.text_list('Format'),
        get_url=_parse_url(el.first_of('DCP/HTTP/Get', 'DCPType/HTTP/Get')),
        post_url=_parse_url(el.first_of('DCP/HTTP/Post', 'DCPType/HTTP/Post')),
        params=params,
    )

    if 'outputFormat' in params:
        op.formats.extend(params['outputFormat'])

    return op


##


def service_metadata(caps_el: gws.IXmlElement) -> dict:
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

    d = _metadata_dict(caps_el.first_of('Service', 'ServiceIdentification'))
    d.update(_contact_dict(caps_el))
    d['contactProviderName'] = caps_el.text_of('ServiceProvider/ProviderName')
    d['contactProviderSite'] = caps_el.text_of('ServiceProvider/ProviderSite')

    #   <Capabilities
    #       <ServiceMetadataURL

    link = _parse_link(caps_el.find('ServiceMetadataURL'))
    if link:
        d['metaLinks'] = [link]

    return gws.strip(d)


def element_metadata(el: gws.IXmlElement) -> dict:
    #   <whatever, e.g. Layer or FeatureType
    #       <Name...
    #       <Title...

    return _metadata_dict(el)


def _metadata_dict(el: gws.IXmlElement) -> dict:
    if not el:
        return {}

    d = {
        'abstract': el.text_of('Abstract'),
        'accessConstraints': el.text_of('AccessConstraints'),
        'attribution': el.text_of('Attribution Title'),
        'fees': el.text_of('Fees'),
        'keywords': el.text_list('Keywords', 'KeywordList', deep=True),
        'name': el.text_of('Name') or el.text_of('Identifier'),
        'title': el.text_of('Title'),
        'metaLinks': gws.compact(_parse_link(e) for e in el.findall('MetadataURL')),
    }

    e = el.find('AuthorityURL')
    if e:
        d['authorityUrl'] = _parse_url(e)
        d['authorityName'] = e.get('name')

    e = el.find('Identifier')
    if e:
        d['authorityIdentifier'] = e.text

    return gws.strip(d)


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


def _contact_dict(caps_el: gws.IXmlElement) -> dict:
    contact_el = caps_el.first_of('Service/ContactInformation', 'ServiceProvider/ServiceContact')
    if not contact_el:
        return {}

    texts = contact_el.text_dict(deep=True)
    d = {}

    for dst, src in _contact_mapping:
        val = texts.get(src, '').strip()
        if val:
            d[dst] = val

    return d


##

def supported_bounds(layer_el: gws.IXmlElement, extra_crsids: t.Optional[t.List[str]] = None) -> t.List[gws.Bounds]:
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

    for el in layer_el.findall('BoundingBox'):
        crs = gws.gis.crs.get(el.get('srs') or el.get('crs'))
        bbox = _parse_bbox(el)
        if crs and bbox:
            crs_to_bounds[crs] = gws.Bounds(crs=crs, extent=bbox)

    # NB prefer these for 4326 to avoid axis issues

    el = layer_el.first_of('EX_GeographicBoundingBox', 'WGS84BoundingBox', 'LatLonBoundingBox')
    if el:
        bbox = _parse_bbox(el)
        if bbox:
            crs = gws.gis.crs.get4326()
            crs_to_bounds[crs] = gws.Bounds(crs=crs, extent=bbox)

    # no bounds

    if not crs_to_bounds:
        return []

    # collect other supported crs and add extras (e.g. wmts matrix sets)

    crsids = set()

    for tag in 'DefaultSRS', 'DefaultCRS', 'OtherSRS', 'OtherCRS', 'SRS', 'CRS':
        for el in layer_el.findall(tag):
            if el.text:
                crsids.add(el.text)

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


def parse_style(el: gws.IXmlElement) -> gws.SourceStyle:
    # <Style>
    #     <Name>default...
    #     <Title>...
    #     <LegendURL
    #         <Format>...
    #         <OnlineResource...

    st = gws.SourceStyle()

    st.metadata = element_metadata(el)
    st.name = st.metadata.get('name', '').lower()
    st.legend_url = _parse_url(el.first_of('LegendURL'))
    st.is_default = (
            el.get('IsDefault') == 'true'
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


def _parse_bbox(el: gws.IXmlElement):
    # note: bboxes are always converted to (x1, y1, x2, y2) with x1 < x2, y1 < y2

    # <BoundingBox/LatLonBoundingBox CRS="..." minx="0" miny="1" maxx="2" maxy="3"/>

    if el.get('minx'):
        return [
            to_float(el.get('minx')),
            to_float(el.get('miny')),
            to_float(el.get('maxx')),
            to_float(el.get('maxy')),
        ]

    # <ows:BoundingBox/WGS84BoundingBox
    #       <ows:LowerCorner> 0 1
    #       <ows:UpperCorner> 2 3

    if el.first_of('LowerCorner'):
        x1, y1 = to_float_pair(el.text_of('LowerCorner'))
        x2, y2 = to_float_pair(el.text_of('UpperCorner'))
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

    if el.first_of('westBoundLongitude'):
        x1 = to_float(el.text_of('eastBoundLongitude'))
        y1 = to_float(el.text_of('southBoundLatitude'))
        x2 = to_float(el.text_of('westBoundLongitude'))
        y2 = to_float(el.text_of('northBoundLatitude'))
        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]


def _parse_url(el: gws.IXmlElement) -> str:
    def cleanup(s):
        return (s or '').strip(' ?&')

    if not el:
        return ''

    # <ows:DCP>
    #       <ows:HTTP>
    #           <ows:Get xlink:href=... <-- we are here

    s = el.get('href') or el.get('onlineResource')
    if s:
        return cleanup(s)

    # <whatever <--
    #       <OnlineResource xlink:href=...

    e = el.first_of('OnlineResource')
    if e:
        return cleanup(e.get('href', default=e.text))

    return ''


def _parse_link(el: gws.IXmlElement) -> t.Optional[gws.MetadataLink]:
    # <MetadataURL type="...
    #       <Format...
    # 	    <OnlineResource...

    if not el:
        return None

    d = gws.strip({
        'url': _parse_url(el),
        'type': el.get('type'),
        'formatName': el.text_of('Format'),
    })

    if d:
        return gws.MetadataLink(d)
