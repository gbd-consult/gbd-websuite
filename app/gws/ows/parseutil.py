import re
import gws
import gws.tools.xml3 as xml3
import gws.types as t


def get_operations(el):
    if _is(el, 'OperationsMetadata'):
        ops = el.all('Operation')
    elif _is(el, 'Capability'):
        ops = el.first('Request').all()
    else:
        return {}

    d = {}

    for e in ops:
        op = t.ServiceOperation()
        op.name = e.attr('name') or e.name
        op.formats = text_list(e, 'Format')
        op.get_url = get_url(one_of(e, 'DCP.HTTP.Get', 'DCPType.HTTP.Get'))
        op.post_url = get_url(one_of(e, 'DCP.HTTP.Post', 'DCPType.HTTP.Post'))
        op.parameters = {
            p.attr('name'): text_list(p, 'Value')
            for p in e.all('Parameter')
        }
        d[op.name] = op

    return d


def get_meta(el):
    if not el:
        return {}

    d = {
        'abstract': compact_ws(el.get_text('Abstract')),
        'access_constraints': check_none(el.get_text('AccessConstraints')),
        'fees': check_none(el.get_text('Fees')),
        'keywords': text_list(el, 'Keywords.Keyword') or text_list(el, 'KeywordList.Keyword'),
        'name': compact_ws(el.get_text('Name') or el.get_text('Identifier')),
        'title': compact_ws(el.get_text('Title')),
        'url': get_url(el.first('MetadataURL')),
    }
    return d


def get_meta_contact(el):
    if not _is(el, 'ContactInformation', 'ServiceProvider'):
        return {}

    texts = extract_text_rec(el)
    d = {}

    for k, v in _contact_mapping:
        v = texts.get(v)
        if v:
            d[k] = v

    return d


def get_extents(el):
    if not el:
        return {}

    d = {}

    for e in el.all('BoundingBox'):
        crs = e.attr('srs') or e.attr('crs')
        if crs:
            d[crs] = _bbox_value(e)

    if 'EPSG:4326' not in d:
        for tag in 'WGS84BoundingBox', 'EX_GeographicBoundingBox', 'LatLonBoundingBox':
            e = el.first(tag)
            if e:
                d['EPSG:4326'] = _bbox_value(e)
                break

    return d


def get_style(el):
    oo = t.SourceStyle()

    oo.meta = t.MetaData(get_meta(el))
    oo.legend = get_url(el.first('LegendURL'))
    oo.is_default = (
            el.get_text('Identifier').lower() == 'default'
            or el.attr('IsDefault') == 'true'
            or oo.meta.name.lower() == 'default')
    return oo


def default_style(styles):
    for s in styles:
        if s.is_default:
            return s
    return styles[0] if styles else None


def crs_from_layers(layers):
    cs = set()

    for sl in layers:
        if not sl.supported_crs:
            continue
        if not cs:
            cs.update(sl.supported_crs)
        else:
            cs = cs.intersection(sl.supported_crs)

    return sorted(cs)


def get_url(el):
    if not el:
        return ''

    # el can be HTTP.Get or HTTP.Post or similar

    p = el.attr('href') or el.attr('onlineResource')
    if p:
        return as_url(p)

    e = el.first('OnlineResource')
    if e:
        return as_url(e.attr('href') or e.text)

    return ''


def one_of(el, *tags):
    for tag in tags:
        e = el.first(tag)
        if e:
            return e


def flatten_source_layers(layers):
    def _collect(ls, res, parent_path, level):
        for sl in ls:
            sl.a_uid = gws.as_uid(sl.name or sl.meta.title)
            sl.a_path = parent_path + '/' + sl.a_uid
            sl.a_level = level
            res.append(sl)
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


def text_list(el, path):
    return gws.compact(e.text.strip() for e in el.all(path))


def check_none(s):
    return '' if s.lower() == 'none' else s


def compact_ws(s):
    return re.sub(r'\s+', ' ', s).strip()


def as_url(s):
    return (s or '').strip(' ?&')


def as_float(s, default=0.0):
    return float(s or default)


def as_int(s, default=0):
    # accept floats as well, but convert to int
    return int(float(s or default))


def as_float_pair(s):
    s = s.split()
    return float(s[0]), float(s[1])


_contact_mapping = [
    # wms

    ('area', 'StateOrProvince'),
    ('city', 'City'),
    ('country', 'Country'),
    ('email', 'ContactElectronicMailAddress'),
    ('fax', 'ContactFacsimileTelephone'),
    ('organization', 'ContactOrganization'),
    ('person', 'ContactPerson'),
    ('phone', 'ContactVoiceTelephone'),
    ('position', 'ContactPosition'),
    ('zip', 'PostCode'),

    # ows

    ('area', 'AdministrativeArea'),
    ('city', 'City'),
    ('country', 'Country'),
    ('email', 'ElectronicMailAddress'),
    ('fax', 'Facsimile'),
    ('organization', 'ProviderName'),
    ('person', 'IndividualName'),
    ('phone', 'Voice'),
    ('position', 'PositionName'),
    ('zip', 'PostalCode'),
]


# note: bboxes are always converted to (x1, y1, x2, y2) with x1 < x2, y1 < y2


def _bbox_value(el):
    # <LatLonBoundingBox minx="-71.63" miny="41.75" maxx="-70.78" maxy="42.90"/>
    if el.attr('minx'):
        return [
            as_float(el.attr('minx')),
            as_float(el.attr('miny')),
            as_float(el.attr('maxx')),
            as_float(el.attr('maxy')),
        ]

    # <ows:BoundingBox>
    # 	<ows:LowerCorner>-20037508.3427892 -20037508.3427892</ows:LowerCorner>
    # 	<ows:UpperCorner>-20037508.3427892 20037508.3427892</ows:UpperCorner>
    # </ows:BoundingBox>
    if el.get('LowerCorner'):
        x1, y1 = as_float_pair(el.get_text('LowerCorner'))
        x2, y2 = as_float_pair(el.get_text('UpperCorner'))
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
        x1 = as_float(el.get_text('eastBoundLongitude')),
        y1 = as_float(el.get_text('southBoundLatitude')),
        x2 = as_float(el.get_text('westBoundLongitude')),
        y2 = as_float(el.get_text('northBoundLatitude')),
        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]


def _is(el, *names):
    if not isinstance(el, xml3.Element):
        return False
    return any(el.name.lower() == n.lower() for n in names)
