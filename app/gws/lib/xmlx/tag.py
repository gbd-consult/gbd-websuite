"""XML builder."""

import re

import gws

from . import element, error, namespace, util


def tag(name: str, *args, **kwargs) -> gws.XmlElement:
    """Build an XML element from arguments.

    Tag and attribute names are local names, Clark names (``{uri}name``) or ``ID:name``, where ``ID`` is the name
    of a well-known namespace (``GML``, ``OWS_11``), which is resolved to a Clark name.

    Example::

        tag('geometry/GML:Point', {'GML:id': 'xy'}, tag('GML:coordinates', '12.345,56.789'), srsName=3857)

    Args:
        name: Tag name, or a slash-separated list of tag names, in which case nested elements are created.
            Slashes inside the ``{uri}`` part of a Clark name do not separate.
        *args: Content of the innermost element. A string, number, bool, date or datetime is appended
            to the text, an ``XmlElement`` is appended as a child, a dict or a ``gws.Data`` object updates
            the attributes (``None`` values are skipped), ``None`` is ignored, and any other iterable
            is processed item by item with the same rules.
        **kwargs: Additional attributes of the innermost element.

    Returns:
        The outermost element.

    Raises:
        BuildError: If a tag name is empty, a namespace ``ID`` is unknown, or an argument cannot be used.
    """

    elements = []

    for part in _split_path(name):
        part = part.strip()
        if not part:
            raise error.BuildError(f'invalid tag name: {name!r}')
        el = element.XmlElement(_resolve_name(part))
        if elements:
            elements[-1].append(el)
        elements.append(el)

    if not elements:
        raise error.BuildError(f'invalid tag name: {name!r}')

    for arg in args:
        _add(elements[-1], arg)

    if kwargs:
        _add(elements[-1], kwargs)

    return elements[0]


##


def _resolve_name(name: str) -> str:
    # local and Clark names are taken as they are, ``ID:name`` is resolved via the well-known table

    uri, prefix, pname = namespace.parse_name(name)
    if uri:
        return name
    if prefix:
        ns = namespace.find_well_known(prefix)
        if not ns:
            raise error.BuildError(f'unknown namespace {prefix!r} in {name!r}')
        return namespace.full_name(pname, ns)
    return name


def _split_path(name: str) -> list[str]:
    # split on '/', but not inside the {uri} part of a Clark name

    return re.split(r'/(?![^{]*})', name)


def _add(el: gws.XmlElement, arg):
    """Add an argument of ``tag()`` to an element."""

    if arg is None:
        return

    if isinstance(arg, gws.XmlElement):
        el.append(arg)
        return

    s, ok = util.atom_to_string(arg)
    if ok:
        _add_text(el, s)
        return

    if gws.is_data_object(arg):
        arg = gws.u.to_dict(arg)

    if isinstance(arg, dict):
        for k, v in arg.items():
            if v is not None:
                el.set(_resolve_name(k), v)
        return

    try:
        arg = list(arg)
    except TypeError:
        raise error.BuildError(f'invalid argument: in {el.tag!r}, {arg=}')

    for a in arg:
        _add(el, a)


def _add_text(el, s):
    """Append text to the element text or to the tail of its last child."""

    if not s:
        return
    if len(el) == 0:
        el.text = (el.text or '') + s
    else:
        el[-1].tail = (el[-1].tail or '') + s
