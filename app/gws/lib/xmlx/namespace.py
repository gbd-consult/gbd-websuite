"""XML namespace manager.

Maintains a registry of XML namespaces (well-known and custom).
"""

from typing import Optional
import re
import os

import gws
from . import error

XMLNS = 'xmlns'


def from_args(**kwargs) -> gws.XmlNamespace:
    """Create a Namespace from keyword arguments."""

    ns = gws.XmlNamespace(**kwargs)
    register(ns)
    return ns


def register(ns: gws.XmlNamespace):
    """Add a Namespace to an internal registry."""

    if ns.uid not in _INDEX.uid:
        _ALL.append(ns)
        _build_index()


def get(uid: str) -> Optional[gws.XmlNamespace]:
    """Locate the Namespace by a uid."""

    return _INDEX.uid.get(uid)


def require(uid: str) -> gws.XmlNamespace:
    """Locate the Namespace by a uid."""

    ns = get(uid)
    if not ns:
        raise error.NamespaceError(f'unknown namespace {uid!r}')
    return ns


def find_by_xmlns(xmlns: str) -> Optional[gws.XmlNamespace]:
    """Locate the Namespace by an xmlns prefix."""

    return _INDEX.xmlns.get(xmlns)


def find_by_uri(uri: str) -> Optional[gws.XmlNamespace]:
    """Locate the Namespace by an Uri."""

    return _INDEX.uri.get(uri)


def split_name(name: str) -> tuple[str, str, str]:
    """Parse an XML name in a xmlns: or Clark notation.

    Args:
        name: XML name.

    Returns:
        A tuple ``(xmlns-prefix, uri, proper name)``.
    """

    if not name:
        return '', '', name

    if name[0] == '{':
        s = name.split('}')
        return '', s[0][1:], s[1]

    if ':' in name:
        s = name.split(':')
        return s[0], '', s[1]

    return '', '', name


def extract(name: str) -> tuple[Optional[gws.XmlNamespace], str]:
    """Extract a Namespace object from a qualified name.

    Args:
        name: XML name.

    Returns:
        A tuple ``(XmlNamespace, proper name)``
    """

    xmlns, uri, pname = split_name(name)

    if xmlns:
        ns = find_by_xmlns(xmlns)
        if not ns:
            raise error.NamespaceError(f'unknown namespace {xmlns!r}')
        return ns, pname

    if uri:
        ns = find_by_uri(uri)
        if not ns:
            raise error.NamespaceError(f'unknown namespace uri {uri!r}')
        return ns, pname

    return None, pname


def qualify_name(name: str, ns: Optional[gws.XmlNamespace] = None, replace: bool = False) -> str:
    """Qualify an XML name.

    Args:
        name: An XML name.
        ns: A namespace.
        replace: If true, replace the existing namespace.

    Returns:
        A qualified name.
    """

    ns2, pname = extract(name)
    if ns2 and not replace:
        return ns2.xmlns + ':' + pname
    if ns:
        return ns.xmlns + ':' + pname
    return pname


def unqualify_name(name: str) -> str:
    """Returns an unqualified XML name."""

    _, _, name = split_name(name)
    return name


def unqualify_default(name: str, default_namespace: gws.XmlNamespace) -> str:
    """Removes the default namespace prefix.

    If the name contains the default namespace, remove it, otherwise return the name as is.

    Args:
        name: An XML name.
        default_namespace: A namespace.

    Returns:
        The name.
    """

    ns, pname = extract(name)
    if ns and ns.uid == default_namespace.uid:
        return pname
    if ns:
        return ns.xmlns + ':' + pname
    return name


def clarkify_name(name: str) -> str:
    """Returns an XML name in the Clark notation.

    Args:
        name: A qualified XML name.

    Returns:
        The XML name in Clark notation.
    """

    ns, pname = extract(name)
    if ns:
        return '{' + ns.uri + '}' + pname
    return pname


def declarations(
    namespaces: dict[str, gws.XmlNamespace],
    with_schema_locations: bool = False,
) -> dict:
    """Returns an xmlns declaration block as dictionary of attributes.

    Args:
        namespaces: Mapping from prefixes to namespaces.
        with_schema_locations: Add the "schemaLocation" attribute.

    Returns:
        A dict of attributes.
    """

    atts = []
    schemas = []

    for xmlns, ns in namespaces.items():
        if xmlns == '':
            atts.append((XMLNS, ns.uri))
        else:
            atts.append((XMLNS + ':' + xmlns, ns.uri))

        if with_schema_locations and ns.schemaLocation:
            schemas.append(ns.uri)
            schemas.append(ns.schemaLocation)

    if schemas:
        atts.append((XMLNS + ':' + _XSI, _XSI_URL))
        atts.append((_XSI + ':schemaLocation', ' '.join(schemas)))

    return dict(sorted(atts))


##


def _collect_namespaces(el: gws.XmlElement, ns_map):
    ns, _ = extract(el.tag)
    if ns:
        ns_map[ns.uid] = ns

    for key in el.attrib:
        ns, _ = extract(key)
        if ns and ns.xmlns != XMLNS:
            ns_map[ns.uid] = ns

    for sub in el:
        _collect_namespaces(sub, ns_map)


def _parse_versioned_uri(uri: str) -> tuple[str, str]:
    m = re.match(r'(.+?)/([\d.]+)$', uri)
    if m:
        return m.group(1), m.group(2)
    return '', uri


_XSI = 'xsi'
_XSI_URL = 'http://www.w3.org/2001/XMLSchema-instance'


_ALL: list[gws.XmlNamespace] = []


class _Index:
    uid = {}
    xmlns = {}
    uri = {}


_INDEX = _Index()


# fake namespace for 'xmlns:'
_ALL.append(
    gws.XmlNamespace(
        uid=XMLNS,
        xmlns=XMLNS,
        uri='',
        schemaLocation='',
        version='',
        isDefault=True,
    )
)


def _load_known():
    def http(u):
        return 'http://' + u if not u.startswith('http') else u

    with open(os.path.dirname(__file__) + '/namespaces.md') as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln.startswith('|'):
                continue
            p = [x.strip() for x in ln.strip('|').split('|')]
            if p[0].startswith('#') or p[0].startswith('-'):
                continue
            uid, xmlns, dflt, version, uri, schema = p
            _ALL.append(
                gws.XmlNamespace(
                    uid=uid,
                    xmlns=xmlns or uid,
                    uri=http(uri),
                    schemaLocation=http(schema) if schema else '',
                    version=version,
                    isDefault=dflt != 'N',
                )
            )


def _build_index():
    _INDEX.uid = {}
    _INDEX.xmlns = {}
    _INDEX.uri = {}

    for ns in _ALL:
        _INDEX.uid[ns.uid] = ns

        if ns.xmlns not in _INDEX.xmlns or ns.isDefault:
            _INDEX.xmlns[ns.xmlns] = ns

        _INDEX.uri[ns.uri] = ns
        if ns.version and not ns.uri.endswith('/' + ns.version):
            _INDEX.uri[ns.uri + '/' + ns.version] = ns


_load_known()
_build_index()
