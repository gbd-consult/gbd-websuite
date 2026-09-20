"""XML namespace helpers.

Provides a table of namespaces and name utilities.

The table contains well-known namespaces (``namespaces.md``), keyed by an uppercase uid (``OWS_11``, ``GML``),
and custom namespaces registered at configuration time (``register``), which have no uid.
A prefix can occur several times (e.g. ``gml`` for GML 2 and GML 3.2). A URI can occur several times
with different schema locations (``GML_2``, ``GML_3_1``); ``find_by_uri`` returns the first row,
a document that needs another one declares it explicitly (``XmlElement.namespaces``).
"""

from typing import Optional
import os

import gws
from . import error

XMLNS = 'xmlns'
"""Prefix of namespace declarations."""

XML = 'xml'
"""Reserved ``xml`` prefix, never declared."""

XML_URI = 'http://www.w3.org/XML/1998/namespace'
"""URI of the reserved ``xml`` prefix."""

ADHOC = 'adhoc:'
"""URI scheme for undeclared prefixes: ``{adhoc:foo}bar`` is written as ``foo:bar`` and never declared."""


def get(uid: str) -> Optional[gws.XmlNamespace]:
    """Locate a well-known Namespace by its uid."""

    return _TABLE.get(uid)


def require(uid: str) -> gws.XmlNamespace:
    """Locate a well-known Namespace by its uid and raise if not found."""

    ns = get(uid)
    if not ns:
        raise error.NamespaceError(f'unknown namespace {uid!r}')
    return ns


def find_by_uri(uri: str) -> Optional[gws.XmlNamespace]:
    """Locate a well-known or registered Namespace by its URI."""

    return _URI_INDEX.get(uri)


def find_by_xmlns(xmlns: str) -> Optional[gws.XmlNamespace]:
    """Locate a well-known or registered Namespace by its prefix, registered ones first."""

    for ns in _CUSTOM:
        if ns.xmlns == xmlns:
            return ns
    for ns in _TABLE.values():
        if ns.xmlns == xmlns:
            return ns
    return None


def register(ns: gws.XmlNamespace):
    """Register a custom namespace.

    Registering the same prefix and URI again is a no-op. A different URI under an existing prefix,
    or an existing URI under a different prefix, is an error.
    """

    old = _URI_INDEX.get(ns.uri)
    if old:
        if old.xmlns == ns.xmlns:
            return
        raise error.NamespaceError(f'namespace {ns.uri!r} is already registered as {old.xmlns!r}')

    old = find_by_xmlns(ns.xmlns)
    if old:
        raise error.NamespaceError(f'namespace prefix {ns.xmlns!r} is already registered for {old.uri!r}')

    _CUSTOM.append(ns)
    _URI_INDEX[ns.uri] = ns


def unregister_all():
    """Remove all custom namespaces (for tests)."""

    for ns in _CUSTOM:
        del _URI_INDEX[ns.uri]
    _CUSTOM.clear()


def new(xmlns: str, uri: str, schemaLocation: str = '', extendsGml: bool = False, uid: str = '') -> gws.XmlNamespace:
    """Create a Namespace object."""

    return gws.XmlNamespace(
        uid=uid,
        xmlns=xmlns,
        uri=uri,
        schemaLocation=schemaLocation,
        extendsGml=extendsGml,
    )


def split_name(name: str) -> tuple[str, str]:
    """Split an XML name into a prefix and a local name.

    Args:
        name: XML name.

    Returns:
        A tuple ``(prefix, local name)``, the prefix is empty if the name is unqualified.
    """

    if ':' in name:
        p, _, n = name.partition(':')
        return p, n
    return '', name


def qualify_name(name: str, ns: Optional[gws.XmlNamespace] = None, replace: bool = False) -> str:
    """Qualify an XML name with the namespace prefix (``gml:Point``), for use in text content.

    Args:
        name: An XML name.
        ns: A namespace.
        replace: If true, replace the existing prefix.

    Returns:
        A qualified name.
    """

    prefix, pname = split_name(name)
    if prefix and not replace:
        return name
    if ns:
        return ns.xmlns + ':' + pname
    return pname


def unqualify_name(name: str) -> str:
    """Returns an unqualified XML name."""

    if name.startswith('{'):
        return split_clark_name(name)[1]
    _, pname = split_name(name)
    return pname


def clark_name(name: str, ns: Optional[gws.XmlNamespace | str] = None) -> str:
    """Create a Clark name (``{uri}name``) from a local name and a namespace or an URI."""

    if not ns:
        return name
    uri = ns if isinstance(ns, str) else ns.uri
    return '{' + uri + '}' + name


def resolve_name(name: str) -> str:
    """Resolve a name for use in an element tree.

    A local or a Clark name is returned as is. ``ID:name``, where ``ID`` is the uid of a well-known namespace,
    is converted to a Clark name. Any other prefixed name is an error.
    """

    if name.startswith('{') or ':' not in name:
        return name
    uid, pname = split_name(name)
    ns = _TABLE.get(uid)
    if not ns:
        raise error.NamespaceError(f'unknown namespace {uid!r} in {name!r}')
    return '{' + ns.uri + '}' + pname


def split_clark_name(name: str) -> tuple[str, str]:
    """Split a Clark name into an URI and a local name.

    Returns:
        A tuple ``(uri, local name)``, the URI is empty if the name is not a Clark name.
    """

    if name.startswith('{'):
        uri, _, pname = name[1:].partition('}')
        return uri, pname
    return '', name


def declarations(
    namespaces: list[gws.XmlNamespace],
    prefixes: Optional[dict[str, str]] = None,
    with_schema_locations: bool = False,
) -> dict:
    """Returns an xmlns declaration block as dictionary of attributes.

    Args:
        namespaces: Namespaces to declare (``xmlns == ''`` is the default namespace).
        prefixes: Mapping from URIs to custom prefixes.
        with_schema_locations: Add the "schemaLocation" attribute.

    Returns:
        A dict of attributes.
    """

    atts = []
    schemas = []
    prefixes = prefixes or {}

    for ns in namespaces:
        xmlns = prefixes.get(ns.uri, ns.xmlns)
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

_XSI = 'xsi'
_XSI_URL = 'http://www.w3.org/2001/XMLSchema-instance'


def _load_table() -> dict[str, gws.XmlNamespace]:
    def http(u):
        return 'http://' + u if not u.startswith('http') else u

    table = {}

    with open(os.path.dirname(__file__) + '/namespaces.md') as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln.startswith('|'):
                continue
            p = [x.strip() for x in ln.strip('|').split('|')]
            if p[0].startswith('#') or p[0].startswith('-'):
                continue
            uid, xmlns, uri, schema = p
            uri = http(uri)
            if uid in table:
                raise error.NamespaceError(f'duplicate namespace uid {uid!r}')
            table[uid] = gws.XmlNamespace(
                uid=uid,
                xmlns=xmlns or uid.lower(),
                uri=uri,
                schemaLocation=http(schema) if schema else '',
                extendsGml=False,
            )

    return table


_TABLE = _load_table()
_CUSTOM: list[gws.XmlNamespace] = []
_URI_INDEX = {}
for _ns in _TABLE.values():
    _URI_INDEX.setdefault(_ns.uri, _ns)


class _Constants:
    def __init__(self):
        for uid, n in _TABLE.items():
            setattr(self, uid, n)


ns = _Constants()
"""Well-known namespaces as constants: ``ns.OWS_11``, ``ns.GML``."""
