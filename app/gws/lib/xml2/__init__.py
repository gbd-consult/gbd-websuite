import xml.parsers.expat
import re

import gws
import gws.types as t


class Error(Exception):
    pass


def from_path(path, keep_ws=False, sort_atts=False, strip_ns=False, to_lower=False) -> gws.XmlElement:
    with open(path, 'rb') as fp:
        inp = fp.read()
    return _Parser().parse(inp, keep_ws, sort_atts, strip_ns, to_lower)


def from_string(inp: t.Union[str, bytes], keep_ws=False, sort_atts=False, strip_ns=False, to_lower=False) -> gws.XmlElement:
    return _Parser().parse(inp, keep_ws, sort_atts, strip_ns, to_lower)


def element(name, attributes=None, children=None, text='', tail='') -> gws.XmlElement:
    el = gws.XmlElement()
    el.name = name
    el.attributes = attributes or {}
    el.children = children or []
    el.text = text or ''
    el.tail = tail or ''
    return el


##


def tag(names: str, *args, **kwargs) -> gws.XmlElement:
    root = el = None

    for name in names.split():
        new = element(name)
        if not el:
            root = new
        else:
            el.children.append(new)
        el = new

    if not el:
        raise Error('invalid name for tag', names)

    for arg in args:
        _tag_add(el, arg)

    _tag_add(el, kwargs)

    return root


def _tag_add(el, arg):
    if arg is None:
        return

    if isinstance(arg, gws.XmlElement):
        el.children.append(arg)
        return

    if isinstance(arg, str):
        if arg:
            _tag_add_text(el, arg)
        return

    if isinstance(arg, (int, float, bool)):
        _tag_add_text(el, str(arg).lower())
        return

    if isinstance(arg, dict):
        for k, v in arg.items():
            if v is not None:
                el.attributes[k] = v
        return

    try:
        ls = arg if isinstance(arg, (list, tuple)) else list(arg)
    except TypeError as exc:
        raise Error('invalid argument for tag', arg) from exc

    if ls and isinstance(ls[0], str):
        el.children.append(tag(*ls))
        return

    for arg2 in ls:
        _tag_add(el, arg2)


def _tag_add_text(el, s):
    if not s:
        return
    if not el.children:
        el.text += s
    else:
        el.children[-1].tail += s


##

def to_string(
    el: gws.XmlElement,
    keep_ws=False,
    sort_atts=False,
    strip_ns=False,
    to_lower=False,
    with_schemas=False,
    with_xml=False,
    with_xmlns=False,
) -> str:
    def text(s):
        if isinstance(s, (int, float, bool)):
            return str(s).lower()
        if not isinstance(s, str):
            s = str(s)
        if not keep_ws:
            s = ' '.join(s.strip().split())
        return encode(s)

    def to_str(el, with_xmlns):
        atts = {}
        for key, val in el.attributes.items():
            if val is None:
                continue
            if strip_ns:
                key = unqualify_name(key)
            if to_lower:
                key = key.lower()
            atts[key] = text(val)

        if with_xmlns:
            atts.update(
                namespaces.declarations(
                    default=el.attributes.get('xmlns'),
                    for_element=el,
                    with_schemas=with_schemas))

        head_pos = len(buf)
        buf.append('')

        s = text(el.text)
        if s:
            buf.append(s)

        for c in el.children:
            to_str(c, False)

        name = el.name
        if strip_ns:
            name = unqualify_name(name)
        if to_lower:
            name = name.lower()

        head = name
        if atts:
            atts_items = atts.items()
            if sort_atts:
                atts_items = sorted(atts_items)
            head += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts_items)

        if len(buf) > head_pos + 1:
            buf[head_pos] = '<' + head + '>'
            buf.append('</' + name + '>')
        else:
            buf[head_pos] += '<' + head + '/>'

        s = text(el.tail)
        if s:
            buf.append(s)

    buf = ['']
    to_str(el, with_xmlns)
    if with_xml:
        buf[0] = _XML_DECL
    return ''.join(buf)


def encode(v) -> str:
    s = str(v)
    s = s.replace("&", "&amp;")
    s = s.replace(">", "&gt;")
    s = s.replace("<", "&lt;")
    s = s.replace('"', "&quot;")
    return s


##


def all(el: t.Optional[gws.XmlElement], *paths) -> t.List[gws.XmlElement]:
    if not el:
        return []
    if not paths:
        return el.children
    els = []
    for path in paths:
        els.extend(_all(el, path))
    return els


def first(el: t.Optional[gws.XmlElement], *paths) -> t.Optional[gws.XmlElement]:
    if not el:
        return None
    if not paths:
        return el.children[0] if el.children else None
    for path in paths:
        els = _all(el, path)
        if els:
            return els[0]


def text(el: t.Optional[gws.XmlElement], path=None) -> str:
    if not el:
        return ''
    if not path:
        return el.text
    els = _all(el, path)
    return els[0].text if els else ''


def text_list(el: t.Optional[gws.XmlElement], *paths, deep=False) -> t.List[str]:
    return _collect_text(el, paths, deep=deep, as_dict=False)


def text_dict(el: t.Optional[gws.XmlElement], *paths, deep=False) -> t.Dict[str, str]:
    return _collect_text(el, paths, deep=deep, as_dict=True)


def iter_all(el: gws.XmlElement):
    if not el:
        return

    yield el

    for c in el.children:
        yield from iter_all(c)


def attr(el: gws.XmlElement, *names, default=None):
    if not el or not el.attributes:
        return default

    for name in names:
        if name in el.attributes:
            return el.attributes[name]

        name = name.lower()
        has_ns = _NSDELIM in name

        for k, v in el.attributes.items():
            if has_ns and k.lower() == name:
                return v
            if not has_ns and unqualify_name(k).lower() == name:
                return v

    return default


def element_is(el: gws.XmlElement, *names) -> bool:
    if not el:
        return False

    for name in names:
        name = name.lower()
        has_ns = _NSDELIM in name

        if has_ns and el.name.lower() == name:
            return True
        if not has_ns and unqualify_name(el.name).lower() == name:
            return True

    return False


def _all(el, path):
    if isinstance(path, str):
        path = path.split()

    els = [el]

    try:
        for name in path:
            index = ''
            if '[' in name:
                name, _, index = name[:-1].partition('[')

            new_els = []

            name = name.lower()
            has_ns = _NSDELIM in name

            for c in els[0].children:
                if has_ns and c.name.lower() == name:
                    new_els.append(c)
                elif not has_ns and unqualify_name(c.name).lower() == name:
                    new_els.append(c)

            els = [new_els[int(index)]] if index else new_els

        return els

    except (KeyError, IndexError, AttributeError):
        return []


def _collect_text(el, paths, deep, as_dict):
    if not el:
        return {} if as_dict else []

    def walk(e):
        if e.text:
            buf.append((e.name, e.text))
        if deep:
            for c in e.children:
                walk(c)

    buf = []

    if not paths:
        walk(el)

    for path in paths:
        for e in _all(el, path):
            walk(e)

    return dict(buf) if as_dict else [t for _, t in buf]


##

class _Parser:
    def parse(
        self,
        inp: t.Union[str, bytes],
        keep_ws,
        sort_atts,
        strip_ns,
        to_lower
    ):

        self.el_stack = []
        self.ns_stack = []
        self.text_buf = []

        self.keep_ws = keep_ws
        self.sort_atts = sort_atts
        self.strip_ns = strip_ns
        self.to_lower = to_lower

        self.el_stack = [element('', children=[])]
        self.ns_stack = [{'': ''}]

        instr = _decode_input(inp)

        self.p = xml.parsers.expat.ParserCreate()
        self.p.buffer_text = True

        self.p.StartElementHandler = self.StartElementHandler
        self.p.EndElementHandler = self.EndElementHandler
        self.p.CharacterDataHandler = self.CharacterDataHandler

        try:
            self.p.Parse(instr, True)
        except xml.parsers.expat.ExpatError as e:
            raise Error('parse error: ' + _XML_Error[getattr(e, 'code')], self.p.ErrorLineNumber, self.p.ErrorColumnNumber)

        for el in self.el_stack[-1].children:
            return el

        raise Error('parse error: no element found')

    def StartElementHandler(self, tag_name, attributes):
        if self.text_buf:
            self._flush_text()

        ns_cur = self.ns_stack[-1]
        ns_new = {}

        atts = {}
        unresolved_atts = []

        for attr_name, val in attributes.items():
            if self.to_lower:
                attr_name = attr_name.lower()

            pfx, name = split_name(attr_name)

            if pfx:
                if pfx == _XMLNS:
                    ns_new[name] = val
                elif self.strip_ns:
                    atts[name] = val
                else:
                    unresolved_atts.append((pfx, name, val))
            else:
                if name == _XMLNS:
                    ns_new[''] = val
                else:
                    atts[name] = val

        for pfx, name, val in unresolved_atts:
            atts[self._qname(pfx, name)] = val

        if self.sort_atts:
            atts = dict(sorted(atts.items()))

        if ns_new:
            ns_cur = dict(ns_cur)
            ns_cur.update(ns_new)
        self.ns_stack.append(ns_cur)

        if self.to_lower:
            tag_name = tag_name.lower()

        pfx, name = split_name(tag_name)
        el = element(name=self._qname(pfx, name), attributes=atts)
        el.pos = [self.p.CurrentLineNumber - 1, self.p.CurrentColumnNumber]

        self.el_stack[-1].children.append(el)
        self.el_stack.append(el)

    def EndElementHandler(self, tag_name):
        if self.text_buf:
            self._flush_text()
        self.el_stack.pop()
        self.ns_stack.pop()

    def CharacterDataHandler(self, data):
        # NB despite `buffer_text` above,
        # this might be called several times in a row on large inputs
        self.text_buf.append(data)

    def _qname(self, pfx, name):
        if self.strip_ns or not pfx:
            return name
        ns = self.ns_stack[-1].get(pfx)
        if ns:
            pfx = namespaces.prefix(ns) or pfx
        return pfx + ':' + name

    def _flush_text(self):
        if len(self.text_buf) == 1:
            text = self.text_buf[0]
        else:
            text = ''.join(self.text_buf)

        self.text_buf = []

        if not self.keep_ws:
            text = ' '.join(text.strip().split())

        if text:
            top = self.el_stack[-1]
            if top.children:
                top.children[-1].tail = text
            else:
                top.text = text


def _decode_input(inp: t.Union[str, bytes]) -> str:
    # the problem is, we can receive a document
    # that is declared ISO-8859-1, but actually is UTF and vice versa.
    # therefore, don't let expat do the decoding, always give it a `str`
    # and remove the xml decl with the (possibly incorrect) encoding

    if isinstance(inp, bytes):
        encodings = []
        inp = inp.strip()
        if inp.startswith(b'<?xml'):
            try:
                end = inp.index(b'?>')
            except ValueError:
                raise Error('invalid XML declaration')

            head = inp[:end].decode('ascii').lower()
            m = re.search(r'encoding\s*=\s*(\S+)', head)
            if m:
                encodings.append(m.group(1).strip('\'\"'))
            inp = inp[end + 2:]

        # declared encoding, if any, first, then utf8, then latin

        if 'utf8' not in encodings:
            encodings.append('utf8')
        if 'iso-8859-1' not in encodings:
            encodings.append('iso-8859-1')

        for enc in encodings:
            try:
                return inp.decode(encoding=enc, errors='strict')
            except (LookupError, UnicodeDecodeError):
                pass

        raise Error(f'invalid encoding, tried {",".join(encodings)}')

    if isinstance(inp, str):
        inp = inp.strip()
        if not inp.startswith('<?xml'):
            return inp
        try:
            end = inp.index('?>')
        except ValueError:
            raise Error('invalid XML declaration')
        return inp[end + 2:]

    raise Error(f'invalid input')


# https://github.com/python/cpython/blob/main/Modules/expat/expat.h

_XML_Error = [
    'XML_ERROR_NONE',
    'XML_ERROR_NO_MEMORY',
    'XML_ERROR_SYNTAX',
    'XML_ERROR_NO_ELEMENTS',
    'XML_ERROR_INVALID_TOKEN',
    'XML_ERROR_UNCLOSED_TOKEN',
    'XML_ERROR_PARTIAL_CHAR',
    'XML_ERROR_TAG_MISMATCH',
    'XML_ERROR_DUPLICATE_ATTRIBUTE',
    'XML_ERROR_JUNK_AFTER_DOC_ELEMENT',
    'XML_ERROR_PARAM_ENTITY_REF',
    'XML_ERROR_UNDEFINED_ENTITY',
    'XML_ERROR_RECURSIVE_ENTITY_REF',
    'XML_ERROR_ASYNC_ENTITY',
    'XML_ERROR_BAD_CHAR_REF',
    'XML_ERROR_BINARY_ENTITY_REF',
    'XML_ERROR_ATTRIBUTE_EXTERNAL_ENTITY_REF',
    'XML_ERROR_MISPLACED_XML_PI',
    'XML_ERROR_UNKNOWN_ENCODING',
    'XML_ERROR_INCORRECT_ENCODING',
    'XML_ERROR_UNCLOSED_CDATA_SECTION',
    'XML_ERROR_EXTERNAL_ENTITY_HANDLING',
    'XML_ERROR_NOT_STANDALONE',
    'XML_ERROR_UNEXPECTED_STATE',
    'XML_ERROR_ENTITY_DECLARED_IN_PE',
    'XML_ERROR_FEATURE_REQUIRES_XML_DTD',
    'XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING',
    'XML_ERROR_UNBOUND_PREFIX',
    'XML_ERROR_UNDECLARING_PREFIX',
    'XML_ERROR_INCOMPLETE_PE',
    'XML_ERROR_XML_DECL',
    'XML_ERROR_TEXT_DECL',
    'XML_ERROR_PUBLICID',
    'XML_ERROR_SUSPENDED',
    'XML_ERROR_NOT_SUSPENDED',
    'XML_ERROR_ABORTED',
    'XML_ERROR_FINISHED',
    'XML_ERROR_SUSPEND_PE',
    'XML_ERROR_RESERVED_PREFIX_XML',
    'XML_ERROR_RESERVED_PREFIX_XMLNS',
    'XML_ERROR_RESERVED_NAMESPACE_URI',
    'XML_ERROR_INVALID_ARGUMENT',
    'XML_ERROR_NO_BUFFER',
    'XML_ERROR_AMPLIFICATION_LIMIT_BREACH',
]

_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
_XMLNS = 'xmlns'
_XSI = 'xsi'
_XSI_URL = 'http://www.w3.org/2001/XMLSchema-instance'

##


_NSDELIM = ':'


##

class Namespaces:
    def __init__(self):
        self._pfx_to_uri = {}
        self._uri_to_pfx = {}
        self._schema = {}
        self._adhoc_ns_count = 0

    def add(self, pfx, uri, schema=''):
        self._pfx_to_uri[pfx] = uri
        self._uri_to_pfx[uri] = pfx
        if schema:
            self._schema[uri] = self._schema[pfx] = schema

    def prefix(self, uri, generate_missing=False):
        pfx = self._uri_to_pfx.get(uri)
        if pfx or not generate_missing:
            return pfx
        self._adhoc_ns_count += 1
        pfx = 'ns' + str(self._adhoc_ns_count)
        self.add(pfx, uri)
        return pfx

    def uri(self, pfx):
        return self._pfx_to_uri.get(pfx)

    def schema(self, uri_or_pfx):
        return self._schema.get(uri_or_pfx)

    def declarations(self, namespaces=None, default=None, with_schemas=True, for_element=None):
        pset = set()

        if for_element:
            _collect_ns_prefixes(for_element, pset)
        if namespaces:
            pset.update(namespaces)
        if default:
            pset.add(default)

        atts = []
        schemas = []

        for pfx in pset:
            uri = self._pfx_to_uri.get(pfx)
            if not uri and pfx in self._uri_to_pfx:
                # ns URI given instead of a prefix?
                uri = pfx
            if not uri:
                raise Error(f'unknown namespace {pfx!r}')
            atts.append((_XMLNS if pfx == default else _XMLNS + ':' + pfx, uri))
            if with_schemas:
                sch = self._schema.get(uri)
                if sch:
                    schemas.append(uri)
                    schemas.append(sch)

        if schemas:
            atts.append((_XMLNS + ':' + _XSI, _XSI_URL))
            atts.append((_XSI + ':schemaLocation', ' '.join(schemas)))

        return dict(sorted(atts))


def _collect_ns_prefixes(el, prefixes):
    pfx, name = split_name(el.name)
    if pfx:
        prefixes.add(pfx)
    for k in el.attributes:
        pfx, name = split_name(k)
        if pfx:
            prefixes.add(name if pfx == _XMLNS else pfx)
    for c in el.children:
        _collect_ns_prefixes(c, prefixes)


def split_name(s):
    if _NSDELIM not in s:
        return '', s
    a, _, b = s.partition(':')
    return a, b


def unqualify_name(s):
    if _NSDELIM not in s:
        return s
    _, _, b = s.partition(':')
    return b


def qualify_name(s, prefix):
    if _NSDELIM not in s:
        return prefix + _NSDELIM + s
    return s


def requalify_name(s, prefix):
    if _NSDELIM not in s:
        return prefix + _NSDELIM + s
    _, _, b = s.partition(':')
    return prefix + _NSDELIM + s


##

# Namespaces used in built-in templates.

# (id, url, schema)

_NAMESPACES = [
    # xml standard

    # NB our templates use "xsd" and "xsi" for XMLSchema namespaces
    ('xsd', 'http://www.w3.org/2001/XMLSchema', ''),
    ('xsi', 'http://www.w3.org/2001/XMLSchema-instance', ''),
    ('xlink', 'http://www.w3.org/1999/xlink', 'https://www.w3.org/XML/2008/06/xlink.xsd'),
    ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns', ''),
    ('soap', 'http://www.w3.org/2003/05/soap-envelope', 'https://www.w3.org/2003/05/soap-envelope/'),

    # ogc

    ('csw', 'http://www.opengis.net/cat/csw/2.0.2', 'http://schemas.opengis.net/csw/2.0.2/csw.xsd'),
    ('dc', 'http://purl.org/dc/elements/1.1/', 'http://schemas.opengis.net/csw/2.0.2/rec-dcmes.xsd'),
    ('dcm', 'http://purl.org/dc/dcmitype/', 'http://dublincore.org/schemas/xmls/qdc/2008/02/11/dcmitype.xsd'),
    ('dct', 'http://purl.org/dc/terms/', 'http://schemas.opengis.net/csw/2.0.2/rec-dcterms.xsd'),
    ('fes', 'http://www.opengis.net/fes/2.0', 'http://schemas.opengis.net/filter/2.0/filterAll.xsd'),
    ('gco', 'http://www.isotc211.org/2005/gco', 'http://schemas.opengis.net/iso/19139/20070417/gco/gco.xsd'),
    ('gmd', 'http://www.isotc211.org/2005/gmd', 'http://schemas.opengis.net/csw/2.0.2/profiles/apiso/1.0.0/apiso.xsd'),
    ('gml', 'http://www.opengis.net/gml/3.2', 'http://schemas.opengis.net/gml/3.2.1/gml.xsd'),
    ('gmlcov', 'http://www.opengis.net/gmlcov/1.0', 'http://schemas.opengis.net/gmlcov/1.0/gmlcovAll.xsd'),
    ('gmx', 'http://www.isotc211.org/2005/gmx', 'http://schemas.opengis.net/iso/19139/20070417/gmx/gmx.xsd'),
    ('ogc', 'http://www.opengis.net/ogc', 'http://schemas.opengis.net/filter/1.1.0/filter.xsd'),
    ('ows', 'http://www.opengis.net/ows/1.1', 'http://schemas.opengis.net/ows/1.0.0/owsAll.xsd'),
    ('sld', 'http://www.opengis.net/sld', 'http://schemas.opengis.net/sld/1.1/sldAll.xsd'),
    ('srv', 'http://www.isotc211.org/2005/srv', 'http://schemas.opengis.net/iso/19139/20070417/srv/1.0/srv.xsd'),
    ('swe', 'http://www.opengis.net/swe/2.0', 'http://schemas.opengis.net/sweCommon/2.0/swe.xsd'),
    ('wcs', 'http://www.opengis.net/wcs/2.0', 'http://schemas.opengis.net/wcs/1.0.0/wcsAll.xsd'),
    ('wcscrs', 'http://www.opengis.net/wcs/crs/1.0', ''),
    ('wcsint', 'http://www.opengis.net/wcs/interpolation/1.0', ''),
    ('wcsscal', 'http://www.opengis.net/wcs/scaling/1.0', ''),
    ('wfs', 'http://www.opengis.net/wfs/2.0', 'http://schemas.opengis.net/wfs/2.0/wfs.xsd'),
    ('wms', 'http://www.opengis.net/wms', 'http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd'),
    ('wmts', 'http://www.opengis.net/wmts/1.0', 'http://schemas.opengis.net/wmts/1.0/wmts.xsd'),

    # inspire

    ('inspire_dls', 'http://inspire.ec.europa.eu/schemas/inspire_dls/1.0', 'http://inspire.ec.europa.eu/schemas/inspire_dls/1.0/inspire_dls.xsd'),
    ('inspire_ds', 'http://inspire.ec.europa.eu/schemas/inspire_ds/1.0', 'http://inspire.ec.europa.eu/schemas/inspire_ds/1.0/inspire_ds.xsd'),
    ('inspire_vs', 'http://inspire.ec.europa.eu/schemas/inspire_vs/1.0', 'http://inspire.ec.europa.eu/schemas/inspire_vs/1.0/inspire_vs.xsd'),
    ('inspire_vs_ows11', 'http://inspire.ec.europa.eu/schemas/inspire_vs_ows11/1.0', 'http://inspire.ec.europa.eu/schemas/inspire_vs_ows11/1.0/inspire_vs_ows_11.xsd'),
    ('inspire_common', 'http://inspire.ec.europa.eu/schemas/common/1.0', 'http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd'),

    # inspire themes

    ('ac-mf', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd'),
    ('ac', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd'),
    ('mf', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd'),
    ('act-core', 'http://inspire.ec.europa.eu/schemas/act-core/4.0', 'http://inspire.ec.europa.eu/schemas/act-core/4.0/ActivityComplex_Core.xsd'),
    ('ad', 'http://inspire.ec.europa.eu/schemas/ad/4.0', 'http://inspire.ec.europa.eu/schemas/ad/4.0/Addresses.xsd'),
    ('af', 'http://inspire.ec.europa.eu/schemas/af/4.0', 'http://inspire.ec.europa.eu/schemas/af/4.0/AgriculturalAndAquacultureFacilities.xsd'),
    ('am', 'http://inspire.ec.europa.eu/schemas/am/4.0', 'http://inspire.ec.europa.eu/schemas/am/4.0/AreaManagementRestrictionRegulationZone.xsd'),
    ('au', 'http://inspire.ec.europa.eu/schemas/au/4.0', 'http://inspire.ec.europa.eu/schemas/au/4.0/AdministrativeUnits.xsd'),
    ('base', 'http://inspire.ec.europa.eu/schemas/base/3.3', 'http://inspire.ec.europa.eu/schemas/base/3.3/BaseTypes.xsd'),
    ('base2', 'http://inspire.ec.europa.eu/schemas/base2/2.0', 'http://inspire.ec.europa.eu/schemas/base2/2.0/BaseTypes2.xsd'),
    ('br', 'http://inspire.ec.europa.eu/schemas/br/4.0', 'http://inspire.ec.europa.eu/schemas/br/4.0/Bio-geographicalRegions.xsd'),
    ('bu-base', 'http://inspire.ec.europa.eu/schemas/bu-base/4.0', 'http://inspire.ec.europa.eu/schemas/bu-base/4.0/BuildingsBase.xsd'),
    ('bu-core2d', 'http://inspire.ec.europa.eu/schemas/bu-core2d/4.0', 'http://inspire.ec.europa.eu/schemas/bu-core2d/4.0/BuildingsCore2D.xsd'),
    ('bu-core3d', 'http://inspire.ec.europa.eu/schemas/bu-core3d/4.0', 'http://inspire.ec.europa.eu/schemas/bu-core3d/4.0/BuildingsCore3D.xsd'),
    ('bu', 'http://inspire.ec.europa.eu/schemas/bu/0.0', 'http://inspire.ec.europa.eu/schemas/bu/0.0/Buildings.xsd'),
    ('cp', 'http://inspire.ec.europa.eu/schemas/cp/4.0', 'http://inspire.ec.europa.eu/schemas/cp/4.0/CadastralParcels.xsd'),
    ('cvbase', 'http://inspire.ec.europa.eu/schemas/cvbase/2.0', 'http://inspire.ec.europa.eu/schemas/cvbase/2.0/CoverageBase.xsd'),
    ('cvgvp', 'http://inspire.ec.europa.eu/schemas/cvgvp/0.1', 'http://inspire.ec.europa.eu/schemas/cvgvp/0.1/CoverageGVP.xsd'),
    ('ef', 'http://inspire.ec.europa.eu/schemas/ef/4.0', 'http://inspire.ec.europa.eu/schemas/ef/4.0/EnvironmentalMonitoringFacilities.xsd'),
    ('el-bas', 'http://inspire.ec.europa.eu/schemas/el-bas/4.0', 'http://inspire.ec.europa.eu/schemas/el-bas/4.0/ElevationBaseTypes.xsd'),
    ('el-cov', 'http://inspire.ec.europa.eu/schemas/el-cov/4.0', 'http://inspire.ec.europa.eu/schemas/el-cov/4.0/ElevationGridCoverage.xsd'),
    ('el-tin', 'http://inspire.ec.europa.eu/schemas/el-tin/4.0', 'http://inspire.ec.europa.eu/schemas/el-tin/4.0/ElevationTin.xsd'),
    ('el-vec', 'http://inspire.ec.europa.eu/schemas/el-vec/4.0', 'http://inspire.ec.europa.eu/schemas/el-vec/4.0/ElevationVectorElements.xsd'),
    ('elu', 'http://inspire.ec.europa.eu/schemas/elu/4.0', 'http://inspire.ec.europa.eu/schemas/elu/4.0/ExistingLandUse.xsd'),
    ('er-b', 'http://inspire.ec.europa.eu/schemas/er-b/4.0', 'http://inspire.ec.europa.eu/schemas/er-b/4.0/EnergyResourcesBase.xsd'),
    ('er-c', 'http://inspire.ec.europa.eu/schemas/er-c/4.0', 'http://inspire.ec.europa.eu/schemas/er-c/4.0/EnergyResourcesCoverage.xsd'),
    ('er-v', 'http://inspire.ec.europa.eu/schemas/er-v/4.0', 'http://inspire.ec.europa.eu/schemas/er-v/4.0/EnergyResourcesVector.xsd'),
    ('er', 'http://inspire.ec.europa.eu/schemas/er/0.0', 'http://inspire.ec.europa.eu/schemas/er/0.0/EnergyResources.xsd'),
    ('gaz', 'http://inspire.ec.europa.eu/schemas/gaz/3.2', 'http://inspire.ec.europa.eu/schemas/gaz/3.2/Gazetteer.xsd'),
    ('ge-core', 'http://inspire.ec.europa.eu/schemas/ge-core/4.0', 'http://inspire.ec.europa.eu/schemas/ge-core/4.0/GeologyCore.xsd'),
    ('ge', 'http://inspire.ec.europa.eu/schemas/ge/0.0', 'http://inspire.ec.europa.eu/schemas/ge/0.0/Geology.xsd'),
    ('ge_gp', 'http://inspire.ec.europa.eu/schemas/ge_gp/4.0', 'http://inspire.ec.europa.eu/schemas/ge_gp/4.0/GeophysicsCore.xsd'),
    ('ge_hg', 'http://inspire.ec.europa.eu/schemas/ge_hg/4.0', 'http://inspire.ec.europa.eu/schemas/ge_hg/4.0/HydrogeologyCore.xsd'),
    ('gelu', 'http://inspire.ec.europa.eu/schemas/gelu/4.0', 'http://inspire.ec.europa.eu/schemas/gelu/4.0/GriddedExistingLandUse.xsd'),
    ('geoportal', 'http://inspire.ec.europa.eu/schemas/geoportal/1.0', 'http://inspire.ec.europa.eu/schemas/geoportal/1.0/geoportal.xsd'),
    ('gn', 'http://inspire.ec.europa.eu/schemas/gn/4.0', 'http://inspire.ec.europa.eu/schemas/gn/4.0/GeographicalNames.xsd'),
    ('hb', 'http://inspire.ec.europa.eu/schemas/hb/4.0', 'http://inspire.ec.europa.eu/schemas/hb/4.0/HabitatsAndBiotopes.xsd'),
    ('hh', 'http://inspire.ec.europa.eu/schemas/hh/4.0', 'http://inspire.ec.europa.eu/schemas/hh/4.0/HumanHealth.xsd'),
    ('hy-n', 'http://inspire.ec.europa.eu/schemas/hy-n/4.0', 'http://inspire.ec.europa.eu/schemas/hy-n/4.0/HydroNetwork.xsd'),
    ('hy-p', 'http://inspire.ec.europa.eu/schemas/hy-p/4.0', 'http://inspire.ec.europa.eu/schemas/hy-p/4.0/HydroPhysicalWaters.xsd'),
    ('hy', 'http://inspire.ec.europa.eu/schemas/hy/4.0', 'http://inspire.ec.europa.eu/schemas/hy/4.0/HydroBase.xsd'),
    ('lc', 'http://inspire.ec.europa.eu/schemas/lc/0.0', 'http://inspire.ec.europa.eu/schemas/lc/0.0/LandCover.xsd'),
    ('lcn', 'http://inspire.ec.europa.eu/schemas/lcn/4.0', 'http://inspire.ec.europa.eu/schemas/lcn/4.0/LandCoverNomenclature.xsd'),
    ('lcr', 'http://inspire.ec.europa.eu/schemas/lcr/4.0', 'http://inspire.ec.europa.eu/schemas/lcr/4.0/LandCoverRaster.xsd'),
    ('lcv', 'http://inspire.ec.europa.eu/schemas/lcv/4.0', 'http://inspire.ec.europa.eu/schemas/lcv/4.0/LandCoverVector.xsd'),
    ('lu', 'http://inspire.ec.europa.eu/schemas/lunom/4.0', 'http://inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd'),
    ('lunom', 'http://inspire.ec.europa.eu/schemas/lunom/4.0', 'http://inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd'),
    ('mr-core', 'http://inspire.ec.europa.eu/schemas/mr-core/4.0', 'http://inspire.ec.europa.eu/schemas/mr-core/4.0/MineralResourcesCore.xsd'),
    ('mu', 'http://inspire.ec.europa.eu/schemas/mu/3.0rc3', 'http://inspire.ec.europa.eu/schemas/mu/3.0rc3/MaritimeUnits.xsd'),
    ('net', 'http://inspire.ec.europa.eu/schemas/net/4.0', 'http://inspire.ec.europa.eu/schemas/net/4.0/Network.xsd'),
    ('nz-core', 'http://inspire.ec.europa.eu/schemas/nz-core/4.0', 'http://inspire.ec.europa.eu/schemas/nz-core/4.0/NaturalRiskZonesCore.xsd'),
    ('nz', 'http://inspire.ec.europa.eu/schemas/nz/0.0', 'http://inspire.ec.europa.eu/schemas/nz/0.0/NaturalRiskZones.xsd'),
    ('of', 'http://inspire.ec.europa.eu/schemas/of/4.0', 'http://inspire.ec.europa.eu/schemas/of/4.0/OceanFeatures.xsd'),
    ('oi', 'http://inspire.ec.europa.eu/schemas/oi/4.0', 'http://inspire.ec.europa.eu/schemas/oi/4.0/Orthoimagery.xsd'),
    ('omop', 'http://inspire.ec.europa.eu/schemas/omop/3.0', 'http://inspire.ec.europa.eu/schemas/omop/3.0/ObservableProperties.xsd'),
    ('omor', 'http://inspire.ec.europa.eu/schemas/omor/3.0', 'http://inspire.ec.europa.eu/schemas/omor/3.0/ObservationReferences.xsd'),
    ('ompr', 'http://inspire.ec.europa.eu/schemas/ompr/3.0', 'http://inspire.ec.europa.eu/schemas/ompr/3.0/Processes.xsd'),
    ('omso', 'http://inspire.ec.europa.eu/schemas/omso/3.0', 'http://inspire.ec.europa.eu/schemas/omso/3.0/SpecialisedObservations.xsd'),
    ('pd', 'http://inspire.ec.europa.eu/schemas/pd/4.0', 'http://inspire.ec.europa.eu/schemas/pd/4.0/PopulationDistributionDemography.xsd'),
    ('pf', 'http://inspire.ec.europa.eu/schemas/pf/4.0', 'http://inspire.ec.europa.eu/schemas/pf/4.0/ProductionAndIndustrialFacilities.xsd'),
    ('plu', 'http://inspire.ec.europa.eu/schemas/plu/4.0', 'http://inspire.ec.europa.eu/schemas/plu/4.0/PlannedLandUse.xsd'),
    ('ps', 'http://inspire.ec.europa.eu/schemas/ps/4.0', 'http://inspire.ec.europa.eu/schemas/ps/4.0/ProtectedSites.xsd'),
    ('sd', 'http://inspire.ec.europa.eu/schemas/sd/4.0', 'http://inspire.ec.europa.eu/schemas/sd/4.0/SpeciesDistribution.xsd'),
    ('selu', 'http://inspire.ec.europa.eu/schemas/selu/4.0', 'http://inspire.ec.europa.eu/schemas/selu/4.0/SampledExistingLandUse.xsd'),
    ('so', 'http://inspire.ec.europa.eu/schemas/so/4.0', 'http://inspire.ec.europa.eu/schemas/so/4.0/Soil.xsd'),
    ('sr', 'http://inspire.ec.europa.eu/schemas/sr/4.0', 'http://inspire.ec.europa.eu/schemas/sr/4.0/SeaRegions.xsd'),
    ('su-core', 'http://inspire.ec.europa.eu/schemas/su-core/4.0', 'http://inspire.ec.europa.eu/schemas/su-core/4.0/StatisticalUnitCore.xsd'),
    ('su-grid', 'http://inspire.ec.europa.eu/schemas/su-grid/4.0', 'http://inspire.ec.europa.eu/schemas/su-grid/4.0/StatisticalUnitGrid.xsd'),
    ('su-vector', 'http://inspire.ec.europa.eu/schemas/su-vector/4.0', 'http://inspire.ec.europa.eu/schemas/su-vector/4.0/StatisticalUnitVector.xsd'),
    ('su', 'http://inspire.ec.europa.eu/schemas/su/0.0', 'http://inspire.ec.europa.eu/schemas/su/0.0/StatisticalUnits.xsd'),
    ('tn-a', 'http://inspire.ec.europa.eu/schemas/tn-a/4.0', 'http://inspire.ec.europa.eu/schemas/tn-a/4.0/AirTransportNetwork.xsd'),
    ('tn-c', 'http://inspire.ec.europa.eu/schemas/tn-c/4.0', 'http://inspire.ec.europa.eu/schemas/tn-c/4.0/CableTransportNetwork.xsd'),
    ('tn-ra', 'http://inspire.ec.europa.eu/schemas/tn-ra/4.0', 'http://inspire.ec.europa.eu/schemas/tn-ra/4.0/RailwayTransportNetwork.xsd'),
    ('tn-ro', 'http://inspire.ec.europa.eu/schemas/tn-ro/4.0', 'http://inspire.ec.europa.eu/schemas/tn-ro/4.0/RoadTransportNetwork.xsd'),
    ('tn-w', 'http://inspire.ec.europa.eu/schemas/tn-w/4.0', 'http://inspire.ec.europa.eu/schemas/tn-w/4.0/WaterTransportNetwork.xsd'),
    ('tn', 'http://inspire.ec.europa.eu/schemas/tn/4.0', 'http://inspire.ec.europa.eu/schemas/tn/4.0/CommonTransportElements.xsd'),
    ('ugs', 'http://inspire.ec.europa.eu/schemas/ugs/0.0', 'http://inspire.ec.europa.eu/schemas/ugs/0.0/UtilityAndGovernmentalServices.xsd'),
    ('us-emf', 'http://inspire.ec.europa.eu/schemas/us-emf/4.0', 'http://inspire.ec.europa.eu/schemas/us-emf/4.0/EnvironmentalManagementFacilities.xsd'),
    ('us-govserv', 'http://inspire.ec.europa.eu/schemas/us-govserv/4.0', 'http://inspire.ec.europa.eu/schemas/us-govserv/4.0/GovernmentalServices.xsd'),
    ('us-net-common', 'http://inspire.ec.europa.eu/schemas/us-net-common/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-common/4.0/UtilityNetworksCommon.xsd'),
    ('us-net-el', 'http://inspire.ec.europa.eu/schemas/us-net-el/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-el/4.0/ElectricityNetwork.xsd'),
    ('us-net-ogc', 'http://inspire.ec.europa.eu/schemas/us-net-ogc/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-ogc/4.0/OilGasChemicalsNetwork.xsd'),
    ('us-net-sw', 'http://inspire.ec.europa.eu/schemas/us-net-sw/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-sw/4.0/SewerNetwork.xsd'),
    ('us-net-tc', 'http://inspire.ec.europa.eu/schemas/us-net-tc/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-tc/4.0/TelecommunicationsNetwork.xsd'),
    ('us-net-th', 'http://inspire.ec.europa.eu/schemas/us-net-th/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-th/4.0/ThermalNetwork.xsd'),
    ('us-net-wa', 'http://inspire.ec.europa.eu/schemas/us-net-wa/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-wa/4.0/WaterNetwork.xsd'),
    ('wfd', 'http://inspire.ec.europa.eu/schemas/wfd/0.0', 'http://inspire.ec.europa.eu/schemas/wfd/0.0/WaterFrameworkDirective.xsd'),
]

namespaces = Namespaces()

for pfx, uri, schema in _NAMESPACES:
    namespaces.add(pfx, uri, schema)
