"""XML builder.

This module provides a single function ``tag``, which creates an Xml Element from a list of arguments.

The first argument to this function is interpreted as a tag name
or a slash separated list of tag names, in which case nested elements are created
(slashes inside the ``{uri}`` part of a Clark name do not separate).

The remaining ``*args`` are interpreted as follows:

- a string, number, bool, date or datetime - appended to the text content of the Element
- an ``XmlElement`` - appended as a child to the Element
- a dict or a ``gws.Data`` object - attributes of the Element are updated from it
- ``None`` - ignored
- any other iterable (list, tuple, generator) - its items are interpreted by the same rules

If keyword arguments are given, they are added to the Element's attributes.

Tag and attribute names are local names, Clark names (``{uri}name``) or ``ID:name``, where ``ID`` is the name
of a well-known namespace (``GML``, ``OWS_11``), which is resolved to a Clark name.

**Example:** ::

    tag('geometry/GML:Point', {'GML:id': 'xy'}, tag('GML:coordinates', '12.345,56.789'), srsName=3857)

creates the following element: ::

    <geometry>
        <gml:Point gml:id="xy" srsName="3857">
            <gml:coordinates>12.345,56.789</gml:coordinates>
        </gml:Point>
    </geometry>

"""

import re

import gws

from . import element, error, namespace, util


def tag(name: str, *args, **kwargs) -> gws.XmlElement:
    """Build an XML element from arguments."""

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
    if not s:
        return
    if len(el) == 0:
        el.text = (el.text or '') + s
    else:
        el[-1].tail = (el[-1].tail or '') + s
