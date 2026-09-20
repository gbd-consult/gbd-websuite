"""XML namespace helpers.

Provides a frozen table of well-known namespaces (``namespaces.md``) and name utilities.
"""

from typing import Optional
import os

import gws
from . import error

XMLNS = 'xmlns'
"""Prefix of namespace declarations."""

XML = 'xml'
"""Reserved ``xml`` prefix, never declared."""


def get(uid: str) -> Optional[gws.XmlNamespace]:
    """Locate a well-known Namespace by its uid."""

    return _TABLE.get(uid)


def require(uid: str) -> gws.XmlNamespace:
    """Locate a well-known Namespace by its uid and raise if not found."""

    ns = get(uid)
    if not ns:
        raise error.NamespaceError(f'unknown namespace {uid!r}')
    return ns


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
    """Qualify an XML name.

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

    _, pname = split_name(name)
    return pname


def declarations(
    namespaces: dict[str, gws.XmlNamespace],
    with_schema_locations: bool = False,
) -> dict:
    """Returns an xmlns declaration block as dictionary of attributes.

    Args:
        namespaces: Mapping from prefixes to namespaces (``''`` is the default namespace).
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
            table[uid] = gws.XmlNamespace(
                uid=uid,
                xmlns=xmlns or uid,
                uri=http(uri),
                schemaLocation=http(schema) if schema else '',
                extendsGml=False,
            )

    return table


_TABLE = _load_table()
