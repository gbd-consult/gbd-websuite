"""Parse utilities for OWS XML files."""

from typing import Optional

import re

import gws
import gws.lib.crs
import gws.lib.extent
import gws.lib.net


def service_operations(caps_el: gws.XmlElement) -> list[gws.OwsOperation]:
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


def _parse_operation(el: gws.XmlElement) -> gws.OwsOperation:
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
        values = param_el.textlist('Value') + param_el.textlist('AllowedValues/Value')
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

    op.postUrl = _parse_url(el.findfirst('DCP/HTTP/Post', 'DCPType/HTTP/Post'))

    u = _parse_url(el.findfirst('DCP/HTTP/Get', 'DCPType/HTTP/Get'))
    op.url, op.params = gws.lib.net.extract_params(u)

    op.formats = el.textlist('Format')
    if 'OUTPUTFORMAT' in op.allowedParameters:
        op.formats.extend(op.allowedParameters['OUTPUTFORMAT'])

    return op


##


def service_metadata(caps_el: gws.XmlElement) -> gws.Metadata:
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

    _element_metadata(caps_el.findfirst('Service', 'ServiceIdentification'), md)
    _contact_metadata(caps_el.findfirst('Service/ContactInformation', 'ServiceProvider/ServiceContact'), md)

    md.contactProviderName = caps_el.textof('ServiceProvider/ProviderName')
    md.contactProviderSite = caps_el.textof('ServiceProvider/ProviderSite')

    #   <Capabilities
    #       <ServiceMetadataURL

    link = _parse_link(caps_el.find('ServiceMetadataURL'))
    if link:
        md.serviceMetaLink = link

    return gws.u.strip(md)


def element_metadata(el: gws.XmlElement) -> gws.Metadata:
    #   <whatever, e.g. Layer or FeatureType
    #       <Name...
    #       <Title...

    md = gws.Metadata()
    _element_metadata(el, md)
    return gws.u.strip(md)


def _element_metadata(el: gws.XmlElement, md: gws.Metadata):
    if not el:
        return

    md.abstract = el.textof('Abstract')
    md.accessConstraints = el.textof('AccessConstraints')
    md.attribution = el.textof('Attribution/Title')
    md.fees = el.textof('Fees')
    md.keywords = el.textlist('Keywords', 'KeywordList', deep=True)
    md.name = el.textof('Name', 'Identifier')
    md.title = el.textof('Title')
    md.metaLinks = gws.u.compact(_parse_link(e) for e in el.findall('MetadataURL'))

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


def _contact_metadata(el: gws.XmlElement, md: gws.Metadata):
    if not el:
        return

    src = el.textdict(deep=True)

    for dkey, skey in _contact_mapping:
        if skey in src:
            setattr(md, dkey, src[skey])


##

def wgs_extent(layer_el: gws.XmlElement) -> Optional[gws.Extent]:
    """Read WGS bounding box from a Layer/FeatureType element.

    Extracts coordinates from ``EX_GeographicBoundingBox`` (WMS), ``WGS84BoundingBox`` (OWS)
    or ``LatLonBoundingBox``. For the latter, assume x=longitude, y=latitude,
    as per OGC 01-068r3, 6.5.6.

    Args:
        layer_el: 'Layer' or 'FeatureType' element.
    """

    el = layer_el.findfirst('EX_GeographicBoundingBox', 'WGS84BoundingBox', 'LatLonBoundingBox')
    if el:
        return gws.lib.extent.from_list(_parse_bbox(el))


def supported_crs(layer_el: gws.XmlElement, extra_crs_ids: list[str] = None) -> list[gws.Crs]:
    """Enumerate supported CRS for a Layer/FeatureType element.

    For WMS, enumerates CRS/SRS and BoundingBox tags,
    for OWS, DefaultCRS and OtherCRS.

    Args:
        layer_el: 'Layer' or 'FeatureType' element.
        extra_crs_ids: additional CRS ids.

    Returns:
        A list of ``Crs`` objects.
    """

    # <Layer...
    #     <CRS>EPSG....
    #     <BoundingBox CRS="EPSG:" minx=....
    #
    # <FeatureType...
    #     <DefaultCRS>urn:ogc:def:crs:EPSG...
    #     <OtherCRS>urn:ogc:def:crs:EPSG...

    crsids = set()

    for el in layer_el.findall('BoundingBox'):
        crsids.add(el.get('SRS') or el.get('CRS'))

    for tag in 'DefaultSRS', 'DefaultCRS', 'OtherSRS', 'OtherCRS', 'SRS', 'CRS':
        for el in layer_el.findall(tag):
            if el.text:
                crsids.add(el.text)

    crsids.update(extra_crs_ids or [])

    return gws.u.compact(gws.lib.crs.get(s) for s in crsids)


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
    st.legendUrl = _parse_url(el.findfirst('LegendURL'))
    st.isDefault = (
            el.get('IsDefault') == 'true'
            or st.name == 'default'
            or st.name.endswith(':default'))
    return st


def default_style(styles: list[gws.SourceStyle]) -> Optional[gws.SourceStyle]:
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


def _parse_bbox(el: gws.XmlElement):
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

    if el.findfirst('LowerCorner'):
        x1, y1 = to_float_pair(el.textof('LowerCorner'))
        x2, y2 = to_float_pair(el.textof('UpperCorner'))
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

    if el.findfirst('westBoundLongitude'):
        x1 = to_float(el.textof('eastBoundLongitude'))
        y1 = to_float(el.textof('southBoundLatitude'))
        x2 = to_float(el.textof('westBoundLongitude'))
        y2 = to_float(el.textof('northBoundLatitude'))
        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]


def _parse_url(el: gws.XmlElement) -> str:
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

    e = el.findfirst('OnlineResource')
    if e:
        return cleanup(e.get('href', default=e.text))

    return ''


def _parse_link(el: gws.XmlElement) -> Optional[gws.MetadataLink]:
    if not el:
        return None

    # see base/ows/server/templatelib.py
    # regarding different MetadataURL formats

    # simple
    if el.get('href'):
        return gws.MetadataLink(url=el.get('href'))

    # nested
    d = gws.u.strip({
        'url': _parse_url(el),
        'type': el.get('type'),
        'format': el.textof('Format'),
    })

    if d:
        return gws.MetadataLink(d)
