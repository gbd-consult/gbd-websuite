"""Namespace table and XML name utilities."""

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
    """Find a well-known namespace by its constant name.

    Args:
        name: Constant name in ``namespace_c``, e.g. ``OWS_11`` or ``GML``.

    Returns:
        The namespace, or ``None`` if there is no such name.
    """

    return _NAME_INDEX.get(name)


def find_by_uri(uri: str) -> Optional[gws.XmlNamespace]:
    """Find a well-known or registered namespace by its URI.

    If several well-known namespaces share the URI, the first one in ``namespace_c`` is returned.

    Args:
        uri: Namespace URI.

    Returns:
        The namespace, or ``None`` if the URI is unknown.
    """

    return _URI_INDEX.get(uri)


def find_by_prefix(prefix: str) -> Optional[gws.XmlNamespace]:
    """Find a well-known or registered namespace by its prefix.

    Registered namespaces are checked first, then the well-known ones in the order of ``namespace_c``.

    Args:
        prefix: Namespace prefix.

    Returns:
        The namespace, or ``None`` if the prefix is unknown.
    """

    for ns in _CUSTOM:
        if ns.prefix == prefix:
            return ns
    for ns in _NAME_INDEX.values():
        if ns.prefix == prefix:
            return ns
    return None


def register(ns: gws.XmlNamespace):
    """Register a custom namespace.

    Registering the same prefix and URI again does nothing.

    Args:
        ns: The namespace to register.

    Raises:
        NamespaceError: If the URI is already known under a different prefix,
            or the prefix is already used for a different URI.
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
    """Remove all registered custom namespaces.

    Used in tests.
    """

    for ns in _CUSTOM:
        del _URI_INDEX[ns.uri]
    _CUSTOM.clear()


def new(prefix: str, uri: str, schemaLocation: str = '', extendsGml: bool = False) -> gws.XmlNamespace:
    """Create a namespace object.

    The namespace is not registered.

    Args:
        prefix: Default prefix, empty for a default namespace declaration.
        uri: Namespace URI.
        schemaLocation: Schema location URL.
        extendsGml: Whether the namespace schema extends the GML schema.

    Returns:
        A new namespace object.
    """

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
    """Create a Clark name (``{uri}name``) from a name and a namespace or a URI.

    An existing prefix or URI in ``name`` is replaced.

    Args:
        name: A local, prefixed or Clark name.
        ns: A namespace object or a URI.

    Returns:
        The Clark name, or the local name if ``ns`` is empty.
    """

    pname = plain_name(name)
    if not ns:
        return pname
    uri = ns if isinstance(ns, str) else ns.uri
    return '{' + uri + '}' + pname


def plain_name(name: str) -> str:
    """Get the local part of an XML name.

    Args:
        name: A local, prefixed or Clark name.

    Returns:
        The local name.
    """

    return parse_name(name)[2]


def declarations(
    namespaces: list[gws.XmlNamespace],
    prefixes: Optional[dict[str, str]] = None,
    with_schema_locations: bool = False,
) -> dict:
    """Create ``xmlns`` declarations as attributes.

    Args:
        namespaces: Namespaces to declare (``prefix == ''`` is the default namespace).
        prefixes: Mapping from URIs to prefixes that replace the default ones.
        with_schema_locations: Add an ``xsi:schemaLocation`` attribute (and the ``xsi`` declaration)
            for namespaces that have a schema location.

    Returns:
        A dict of attribute names and values, sorted by name.
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
