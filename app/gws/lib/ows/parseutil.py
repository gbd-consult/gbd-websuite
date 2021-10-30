import re

import gws
import gws.lib.gis
import gws.lib.metadata
import gws.lib.proj
import gws.lib.xml2 as xml2
import gws.types as t


def get_operations(root_el) -> t.List[gws.OwsOperation]:
    els = root_el.all('OperationsMetadata.Operation')
    if els:
        return [get_operation(e) for e in els]

    el = root_el.first('Capability.Request')
    if el:
        return [get_operation(e) for e in el.all()]

    return []


def get_operation(el) -> gws.OwsOperation:
    params = {}
    for p in el.all('Parameter'):
        values = text_list(p, 'Value', 'AllowedValues.Value')
        if values:
            params[p.attr('name')] = values

    op = gws.OwsOperation(
        verb=el.attr('name') or el.name,
        formats=text_list(el, 'Format'),
        get_url=get_url(one_of(el, 'DCP.HTTP.Get', 'DCPType.HTTP.Get')),
        post_url=get_url(one_of(el, 'DCP.HTTP.Post', 'DCPType.HTTP.Post')),
        params=params,
    )

    if 'outputFormat' in params:
        op.formats.extend(params['outputFormat'])

    return op


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


def get_service_metadata(root_el) -> gws.lib.metadata.Metadata:
    md = get_metadata(one_of(root_el, 'Service', 'ServiceIdentification'))

    el = one_of(root_el, 'Service.ContactInformation', 'ServiceProvider.ServiceContact')
    if el:
        texts = extract_text_rec(el)
        contact = {}
        for dst, src in _contact_mapping:
            contact[dst] = texts.get(src)
        md.extend(gws.strip(contact))

    ml = get_link(root_el.first('ServiceMetadataURL'))
    if ml:
        md.extend({
            'metaLinks': [ml]
        })

    return md


def get_metadata(el) -> gws.lib.metadata.Metadata:
    if not el:
        return gws.lib.metadata.from_dict({})

    d = {
        'abstract': compact_ws(el.get_text('Abstract')),
        'accessConstraints': el.get_text('AccessConstraints'),
        'attribution': compact_ws(el.get_text('Attribution.Title')),
        'fees': el.get_text('Fees'),
        'keywords': text_list(el, 'Keywords.Keyword', 'KeywordList.Keyword'),
        'name': compact_ws(el.get_text('Name') or el.get_text('Identifier')),
        'title': compact_ws(el.get_text('Title')),
        'metaLinks': gws.compact(get_link(e) for e in el.all('MetadataURL')),
    }

    e = el.first('AuthorityURL')
    if e:
        d['authorityUrl'] = get_url(e)
        d['authorityName'] = e.attr('name')

    e = el.first('Identifier')
    if e:
        d['authorityIdentifier'] = e.text

    return gws.lib.metadata.from_dict(gws.strip(d))


def get_bounds_list(el) -> t.List[gws.Bounds]:
    if not el:
        return []

    d = {}

    for e in el.all('BoundingBox'):
        crs = e.attr('srs') or e.attr('crs')
        if crs:
            proj = gws.lib.proj.to_proj(crs)
            if proj:
                d[proj.epsg] = _bbox_value(e)

    # NB prefer these for 4326 to avoid axis issues
    e = one_of(el, 'EX_GeographicBoundingBox', 'WGS84BoundingBox', 'LatLonBoundingBox')
    if e:
        d[gws.EPSG_4326] = _bbox_value(e)

    return [gws.Bounds(crs=k, extent=v) for k, v in d.items()]


def get_style(el) -> gws.lib.gis.SourceStyle:
    st = gws.lib.gis.SourceStyle()

    st.metadata = get_metadata(el)
    st.name = st.metadata.get('name', '').lower()
    st.legend_url = get_url(el.first('LegendURL'))
    st.is_default = (
            el.get_text('Identifier').lower() == 'default'
            or el.attr('IsDefault') == 'true'
            or st.name == 'default')
    return st


def default_style(styles) -> t.Optional[gws.lib.gis.SourceStyle]:
    for s in styles:
        if s.is_default:
            return s
    return styles[0] if styles else None


def get_link(el) -> t.Optional[gws.lib.metadata.Link]:
    # el is a MetadataURL element

    if not el:
        return None

    d = gws.strip({
        'url': get_url(el),
        'type': el.attr('type'),
        'formatName': el.get_text('Format'),
    })

    if d:
        return gws.lib.metadata.Link(d)


def get_url(el) -> str:
    # el can be HTTP.Get or HTTP.Post or similar

    if not el:
        return ''

    p = el.attr('href') or el.attr('onlineResource')
    if p:
        return to_url(p)

    e = el.first('OnlineResource')
    if e:
        return to_url(e.attr('href') or e.text)

    return ''


def one_of(el, *tags):
    for tag in tags:
        e = el.first(tag)
        if e:
            return e


def flatten_source_layers(layers):
    def _collect(ls, res, parent_path, level):
        for sl in ls:
            if not sl:
                continue
            sl.a_uid = gws.to_uid(sl.name or sl.metadata.get('title'))
            sl.a_path = parent_path + '/' + sl.a_uid
            sl.a_level = level
            res.append(sl)
            if sl.layers:
                _collect(sl.layers, res, sl.a_path, level + 1)
        return res

    return _collect(layers, [], '', 1)


def extract_text_rec(el):
    if not el:
        return {}

    d = {}

    if el.text and not d.get(el.name):
        d[el.name] = el.text

    for e in el.all():
        d.update(extract_text_rec(e))

    return d


def text_list(el, *paths):
    for path in paths:
        ls = el.all(path)
        if ls:
            return gws.strip(e.text for e in ls)
    return []


def compact_ws(s):
    return re.sub(r'\s+', ' ', s).strip()


def to_url(s):
    return (s or '').strip(' ?&')


def to_float(s, default=0.0):
    return float(s or default)


def to_int(s, default=0):
    # accept floats as well, but convert to int
    return int(float(s or default))


def to_float_pair(s):
    s = s.split()
    return float(s[0]), float(s[1])


# note: bboxes are always converted to (x1, y1, x2, y2) with x1 < x2, y1 < y2


def _bbox_value(el):
    # <LatLonBoundingBox minx="-71.63" miny="41.75" maxx="-70.78" maxy="42.90"/>
    if el.attr('minx'):
        return [
            to_float(el.attr('minx')),
            to_float(el.attr('miny')),
            to_float(el.attr('maxx')),
            to_float(el.attr('maxy')),
        ]

    # <ows:BoundingBox>
    # 	<ows:LowerCorner>-20037508.3427892 -20037508.3427892</ows:LowerCorner>
    # 	<ows:UpperCorner>-20037508.3427892 20037508.3427892</ows:UpperCorner>
    # </ows:BoundingBox>
    if el.get('LowerCorner'):
        x1, y1 = to_float_pair(el.get_text('LowerCorner'))
        x2, y2 = to_float_pair(el.get_text('UpperCorner'))
        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]

    # <EX_GeographicBoundingBox>
    #     <westBoundLongitude>-71.63</westBoundLongitude>
    #     <eastBoundLongitude>-70.78</eastBoundLongitude>
    #     <southBoundLatitude>41.75</southBoundLatitude>
    #     <northBoundLatitude>42.90</northBoundLatitude>
    # </EX_GeographicBoundingBox>
    if el.get('westBoundLongitude'):
        x1 = to_float(el.get_text('eastBoundLongitude'))
        y1 = to_float(el.get_text('southBoundLatitude'))
        x2 = to_float(el.get_text('westBoundLongitude'))
        y2 = to_float(el.get_text('northBoundLatitude'))
        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]


def _is(el, *names):
    if not isinstance(el, xml2.Element):
        return False
    return any(el.name.lower() == n.lower() for n in names)
