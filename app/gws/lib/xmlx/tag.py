"""XML builder.

This module provides a single function ``tag``, which creates an Xml Element from a list of arguments.

The first argument to this function is interpreted as a tag name
or a slash separated list of tag names, in which case nested elements are created.

The remaining ``*args`` are interpreted as follows:

- a simple string or number value - appended to the text content of the Element
- an `XmlElement` - appended as a child to the Element
- a dict - attributes of the Element are updated from this dict
- a list, tuple or a generator - used as arguments to ``tag`` to create a child tag

If keyword arguments are given, they are added to the Element's attributes.

**Example:** ::

    tag('geometry/gml:Point', {'gml:id': 'xy'}, ['gml:coordinates', '12.345,56.789'], srsName=3857)

creates the following element: ::

    <geometry>
        <gml:Point gml:id="xy" srsName="3857">
            <gml:coordinates>12.345,56.789</gml:coordinates>
        </gml:Point>
    </geometry>

"""

import re

import gws

from . import element, error, util


def tag(name: str, *args, **kwargs) -> gws.XmlElement:
    """Build an XML element from arguments."""

    els = []

    for n in _split_name(name):
        el = element.XmlElementImpl(n.strip())
        if not els:
            els.append(el)
        else:
            els[-1].append(el)
            els.append(el)

    if not els:
        raise error.BuildError(f'invalid tag name: {name!r}')

    for arg in args:
        _add(els[-1], arg)

    if kwargs:
        _add(els[-1], kwargs)

    return els[0]


##


def _add(el: gws.XmlElement, arg):
    if arg is None:
        return

    if isinstance(arg, element.XmlElementImpl):
        el.append(arg)
        return

    s, ok = util.atom_to_string(arg)
    if ok:
        _add_text(el, s)
        return

    if isinstance(arg, dict):
        for k, v in arg.items():
            if v is not None:
                el.set(k, v)
        return

    if isinstance(arg, (list, tuple)):
        _add_list(el, arg)
        return

    try:
        ls = list(arg)
    except Exception as exc:
        raise error.BuildError(f'invalid argument: in {el.tag!r}, {arg=}') from exc

    _add_list(el, ls)


def _add_text(el, s):
    if not s:
        return
    if len(el) == 0:
        el.text = (el.text or '') + s
    else:
        el[-1].tail = (el[-1].tail or '') + s


def _add_list(el, ls):
    if not ls:
        return
    if isinstance(ls[0], str):
        _add(el, tag(*ls))
        return
    for arg in ls:
        _add(el, arg)


def _split_name(name):
    if '{' not in name:
        return [s.strip() for s in name.split('/')]

    parts = []
    ns = ''

    for n, s in re.findall(r'({.+?})|([^/{}]+)', name):
        if n:
            ns = n
        else:
            s = s.strip()
            if s:
                parts.append(ns + s)
                ns = ''

    return parts
