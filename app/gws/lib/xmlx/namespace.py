"""XML namespace helpers.

Provides a table of namespaces and name utilities.

The table contains well-known namespaces (``namespace_c.py``, accessible as ``namespace.c.<NAME>``),
keyed by an uppercase name (``OWS_11``, ``GML``), and custom namespaces registered at configuration time (``register``),
which have no name.
A prefix can occur several times (e.g. ``gml`` for GML 2 and GML 3.2). A URI can occur several times
with different schema locations (``GML_2``, ``GML_3_1``); ``find_by_uri`` returns the first row,
a document that needs another one declares it explicitly (``XmlElement.namespaces``).
"""

from typing import Optional

import gws
from . import error
from . import namespace_c as c

XMLNS = 'xmlns'
"""Prefix of namespace declarations."""

XML = 'xml'
"""Reserved ``xml`` prefix, never declared."""

XML_URI = 'http://www.w3.org/XML/1998/namespace'
"""URI of the reserved ``xml`` prefix."""

ADHOC = 'adhoc:'
"""URI scheme for undeclared prefixes: ``{adhoc:foo}bar`` is written as ``foo:bar`` and never declared."""


def find_well_known(name: str) -> Optional[gws.XmlNamespace]:
    """Locate a well-known Namespace by its constant name (``OWS_11``, ``GML``)."""

    return _NAME_INDEX.get(name)


def find_by_uri(uri: str) -> Optional[gws.XmlNamespace]:
    """Locate a well-known or registered Namespace by its URI."""

    return _URI_INDEX.get(uri)


def find_by_prefix(prefix: str) -> Optional[gws.XmlNamespace]:
    """Locate a well-known or registered Namespace by its prefix, registered ones first."""

    for ns in _CUSTOM:
        if ns.prefix == prefix:
            return ns
    for ns in _NAME_INDEX.values():
        if ns.prefix == prefix:
            return ns
    return None


def register(ns: gws.XmlNamespace):
    """Register a custom namespace.

    Registering the same prefix and URI again is a no-op. A different URI under an existing prefix,
    or an existing URI under a different prefix, is an error.
    """

    old = _URI_INDEX.get(ns.uri)
    if old:
        if old.prefix == ns.prefix:
            return
        raise error.NamespaceError(f'namespace {ns.uri!r} is already registered as {old.prefix!r}')

    old = find_by_prefix(ns.prefix)
    if old:
        raise error.NamespaceError(f'namespace prefix {ns.prefix!r} is already registered for {old.uri!r}')

    _CUSTOM.append(ns)
    _URI_INDEX[ns.uri] = ns


def unregister_all():
    """Remove all custom namespaces (for tests)."""

    for ns in _CUSTOM:
        del _URI_INDEX[ns.uri]
    _CUSTOM.clear()


def new(prefix: str, uri: str, schemaLocation: str = '', extendsGml: bool = False) -> gws.XmlNamespace:
    """Create a Namespace object."""

    return gws.XmlNamespace(
        prefix=prefix,
        uri=uri,
        schemaLocation=schemaLocation,
        extendsGml=extendsGml,
    )


def parse_name(name: str) -> tuple[str, str, str]:
    """Parse an XML name.

    Args:
        name: A local (``foo``), a prefixed (``p:foo``) or a Clark (``{uri}foo``) name.

    Returns:
        A tuple ``(uri, prefix, local name)``; ``uri`` and ``prefix`` are empty when not present.
    """

    if name.startswith('{'):
        uri, _, pname = name[1:].partition('}')
        return uri, '', pname
    if ':' in name:
        prefix, _, pname = name.partition(':')
        return '', prefix, pname
    return '', '', name


def full_name(name: str, ns: Optional[gws.XmlNamespace | str]) -> str:
    """Create a Clark name (``{uri}name``) from a name and a namespace or an URI.

    An existing prefix or URI in ``name`` is replaced. Without a namespace, the local name is returned.
    """

    pname = plain_name(name)
    if not ns:
        return pname
    uri = ns if isinstance(ns, str) else ns.uri
    return '{' + uri + '}' + pname


def plain_name(name: str) -> str:
    """Returns the local part of an XML name."""

    return parse_name(name)[2]


def declarations(
    namespaces: list[gws.XmlNamespace],
    prefixes: Optional[dict[str, str]] = None,
    with_schema_locations: bool = False,
) -> dict:
    """Returns an xmlns declaration block as dictionary of attributes.

    Args:
        namespaces: Namespaces to declare (``prefix == ''`` is the default namespace).
        prefixes: Mapping from URIs to custom prefixes.
        with_schema_locations: Add the "schemaLocation" attribute.

    Returns:
        A dict of attributes.
    """

    atts = []
    schemas = []
    prefixes = prefixes or {}

    for ns in namespaces:
        prefix = prefixes.get(ns.uri, ns.prefix)
        if prefix == '':
            atts.append((XMLNS, ns.uri))
        else:
            atts.append((XMLNS + ':' + prefix, ns.uri))

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


_CUSTOM: list[gws.XmlNamespace] = []

_NAME_INDEX = {}
_URI_INDEX = {}

for _name, _ns in vars(c).items():
    if isinstance(_ns, gws.XmlNamespace):
        _NAME_INDEX.setdefault(_name, _ns)
        _URI_INDEX.setdefault(_ns.uri, _ns)
