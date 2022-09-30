"""Parse utilities for OWS XML files."""

import re

import gws
import gws.gis.crs
import gws.gis.extent
import gws.lib.net

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
    op = gws.OwsOperation(verb=el.get('name') or el.tag)

    # @TODO Range
    # @TODO Constraint

    # <Parameter name="Format">
    #   <AllowedValues>
    #       <Value>image/gif</Value>
    # ...

    # <Parameter name="AcceptVersions">
    #    <Value>1.0.0</Value>

    op.allowedParameters = {}
    for param_el in el.findall('Parameter'):
        values = param_el.text_list('Value') + param_el.text_list('AllowedValues/Value')
        if values:
            op.allowedParameters[param_el.get('name').upper()] = values

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

    op.postUrl = _parse_url(el.first_of('DCP/HTTP/Post', 'DCPType/HTTP/Post'))

    u = _parse_url(el.first_of('DCP/HTTP/Get', 'DCPType/HTTP/Get'))
    op.url, op.params = gws.lib.net.extract_params(u)

    op.formats = el.text_list('Format')
    if 'outputFormat' in op.allowedParameters:
        op.formats.extend(op.allowedParameters['outputFormat'])

    return op


##


def service_metadata(caps_el: gws.IXmlElement) -> gws.Metadata:
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

    md = gws.Metadata()

    _element_metadata(caps_el.first_of('Service', 'ServiceIdentification'), md)
    _contact_metadata(caps_el.first_of('Service/ContactInformation', 'ServiceProvider/ServiceContact'), md)

    md.contactProviderName = caps_el.text_of('ServiceProvider/ProviderName')
    md.contactProviderSite = caps_el.text_of('ServiceProvider/ProviderSite')

    #   <Capabilities
    #       <ServiceMetadataURL

    link = _parse_link(caps_el.find('ServiceMetadataURL'))
    if link:
        md.serviceMetaLink = link

    return gws.strip(md)


def element_metadata(el: gws.IXmlElement) -> gws.Metadata:
    #   <whatever, e.g. Layer or FeatureType
    #       <Name...
    #       <Title...

    md = gws.Metadata()
    _element_metadata(el, md)
    return gws.strip(md)


def _element_metadata(el: gws.IXmlElement, md: gws.Metadata):
    if not el:
        return

    md.abstract = el.text_of('Abstract')
    md.accessConstraints = el.text_of('AccessConstraints')
    md.attribution = el.text_of('Attribution/Title')
    md.fees = el.text_of('Fees')
    md.keywords = el.text_list('Keywords', 'KeywordList', deep=True)
    md.name = el.text_of('Name', 'Identifier')
    md.title = el.text_of('Title')
    md.metaLinks = gws.compact(_parse_link(e) for e in el.findall('MetadataURL'))

    e = el.find('AuthorityURL')
    if e:
        md.authorityUrl = _parse_url(e)
        md.authorityName = e.get('name')

    e = el.find('Identifier')
    if e:
        md.authorityIdentifier = e.text


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


def _contact_metadata(el: gws.IXmlElement, md: gws.Metadata):
    if not el:
        return

    src = el.text_dict(deep=True)

    for dkey, skey in _contact_mapping:
        if skey in src:
            setattr(md, dkey, src[skey])


##

def bounds_and_crs(
        layer_el: gws.IXmlElement,
        extra_crsids: t.Optional[t.List[str]] = None
):
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

    wgs_ext = None

    el = layer_el.first_of('EX_GeographicBoundingBox', 'WGS84BoundingBox', 'LatLonBoundingBox')
    if el:
        wgs_ext = _parse_bbox(el)

    bounds = []

    for el in layer_el.findall('BoundingBox'):
        crsid = el.get('SRS') or el.get('CRS')
        fmt, crs = gws.gis.crs.parse(crsid)
        ext = _parse_bbox(el)
        if crs and ext:
            bounds.append(gws.SourceBounds(crs=crs, format=fmt, extent=ext))

    crsids = set(b.crs.srid for b in bounds)

    for tag in 'DefaultSRS', 'DefaultCRS', 'OtherSRS', 'OtherCRS', 'SRS', 'CRS':
        for el in layer_el.findall(tag):
            if el.text:
                crsids.add(el.text)

    if extra_crsids:
        crsids.update(extra_crsids)

    crs_list = gws.compact(gws.gis.crs.get(s) for s in crsids)

    return bounds, crs_list, wgs_ext


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
    st.legendUrl = _parse_url(el.first_of('LegendURL'))
    st.isDefault = (
            el.get('IsDefault') == 'true'
            or st.name == 'default'
            or st.name.endswith(':default'))
    return st


def default_style(styles: t.List[gws.SourceStyle]) -> t.Optional[gws.SourceStyle]:
    for s in styles:
        if s.isDefault:
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
        'format': el.text_of('Format'),
    })

    if d:
        return gws.MetadataLink(d)
