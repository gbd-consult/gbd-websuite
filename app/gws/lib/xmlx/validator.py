"""Schema validator."""

import re
import os
import lxml.etree
import requests

import gws


class Error(gws.Error):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = args[0]
        self.linenoe = args[1]


def validate(xml: str | bytes):
    try:
        parser = lxml.etree.XMLParser(resolve_entities=True)
        parser.resolvers.add(_CachingResolver())

        schema_locations = _extract_schema_locations(xml)
        xsd = _create_combined_xsd(schema_locations)

        xml_tree = _etree(xml, parser)
        schema_tree = _etree(xsd, parser)
        schema = lxml.etree.XMLSchema(schema_tree)
    except lxml.etree.Error as exc:
        raise _error(exc) from exc

    try:
        schema.assertValid(xml_tree)
        return True
    except Exception as exc:
        raise _error(exc) from exc


def _extract_schema_locations(xml: str | bytes) -> dict:
    tree = _etree(xml, None)
    root = tree.getroot()

    xsi_ns = '{http://www.w3.org/2001/XMLSchema-instance}'
    attr = root.get(f'{xsi_ns}schemaLocation')
    if not attr:
        attr = root.get('schemaLocation')
    if not attr:
        return {}

    d = {}

    parts = attr.strip().split()
    while parts:
        namespace = parts.pop(0)
        location = parts.pop(0)
        d[namespace] = location

    return d


def _create_combined_xsd(schema_locations: dict) -> str:
    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">')

    for ns, loc in schema_locations.items():
        xml.append(f'<xs:import namespace="{ns}" schemaLocation="{loc}"/>')

    xml.append('</xs:schema>\n')

    return '\n'.join(xml)


def _etree(xml: str | bytes, parser: lxml.etree.XMLParser | None) -> lxml.etree.ElementTree:
    if isinstance(xml, str):
        xml = xml.encode('utf-8')
    return lxml.etree.ElementTree(lxml.etree.fromstring(xml, parser))


def _error(exc):
    # exc is either {'message': ..., 'lineno': ...}
    # or {'error_log': '<string>:17:0:ERROR:...}

    cls = exc.__class__.__name__

    s = getattr(exc, 'error_log', None)
    if s:
        try:
            lineno = int(s.split(':')[1])
        except Exception:
            lineno = 0
        return Error(f'{cls}: {s}', lineno)

    lineno = getattr(exc, 'lineno', 0)
    return Error(f'{cls}: {exc}', lineno)


class _CachingResolver(lxml.etree.Resolver):
    def resolve(self, url, id, context):
        if url.startswith(('http://', 'https://')):
            if '.loc' in url or 'local' in url:
                buf = _download_url(url, with_cache=False)
            else:
                buf = _download_url(url, with_cache=True)
            return self.resolve_string(buf, context, base_url=url)

        return super().resolve(url, id, context)


def _download_url(url: str, with_cache: bool) -> bytes:
    if not with_cache:
        return _raw_download_url(url)

    cache_dir = gws.u.ensure_dir(gws.c.CACHE_DIR + '/xmlx')
    cache_path = _cache_path(cache_dir, url)

    if os.path.exists(cache_path):
        return gws.u.read_file_b(cache_path)

    content = _raw_download_url(url)
    gws.u.write_file_b(cache_path, content)
    return content


def _raw_download_url(url: str) -> bytes:
    gws.log.debug(f'xmlx.validator: downloading {url!r}')
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(f'Failed to download {url!r}: {response.status_code}')
    return response.content


def _cache_path(cache_dir: str, url: str) -> str:
    u = url.strip().split('//')[-1]
    if '?' in u:
        u = u.split('?', 1)[0]
    fname = 'index.xml'
    parts = u.split('/')

    if u.endswith('/'):
        parts.pop()
    else:
        m = re.search(r'[^/]+\.[a-z]+$', parts[-1])
        if m:
            fname = m.group(0)
            parts.pop()

    d = '/'.join(_to_dirname(p) for p in parts)
    if not d:
        return cache_dir + '/' + fname
    d = gws.u.ensure_dir(cache_dir + '/' + d)
    return d + '/' + fname


def _to_dirname(s: str) -> str:
    s = s.lower().strip().lstrip('.')
    s = re.sub(r'[^a-zA-Z0-9.]+', '_', s).strip('_')
    return s
