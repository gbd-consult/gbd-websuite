"""XML builder.

This module provides a single function ``tag``, which creates an Xml Element from a list of arguments.

The first argument to this function is interpreted as a tag name
or a slash separated list of tag names, in which case nested elements are created.

The remaining ``*args`` are interpreted as follows:

- a string, number, bool, date or datetime - appended to the text content of the Element
- an ``XmlElement`` - appended as a child to the Element
- a dict - attributes of the Element are updated from this dict
- ``None`` - ignored
- any other iterable (list, tuple, generator) - its items are interpreted by the same rules

If keyword arguments are given, they are added to the Element's attributes.

**Example:** ::

    tag('geometry/gml:Point', {'gml:id': 'xy'}, tag('gml:coordinates', '12.345,56.789'), srsName=3857)

creates the following element: ::

    <geometry>
        <gml:Point gml:id="xy" srsName="3857">
            <gml:coordinates>12.345,56.789</gml:coordinates>
        </gml:Point>
    </geometry>

"""

import collections.abc

import gws

from . import element, error, util


def tag(name: str, *args, **kwargs) -> gws.XmlElement:
    """Build an XML element from arguments."""

    stack = []

    for n in name.split('/'):
        n = n.strip()
        if not n:
            raise error.BuildError(f'invalid tag name: {name!r}')
        el = element.XmlElement(n)
        if stack:
            stack[-1].append(el)
        stack.append(el)

    if not stack:
        raise error.BuildError(f'invalid tag name: {name!r}')

    for arg in args:
        _add(stack[-1], arg)

    if kwargs:
        _add(stack[-1], kwargs)

    return stack[0]


##


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

    if isinstance(arg, dict):
        for k, v in arg.items():
            if v is not None:
                el.set(k, v)
        return

    if isinstance(arg, collections.abc.Iterable) and not isinstance(arg, gws.Data):
        for a in arg:
            _add(el, a)
        return

    raise error.BuildError(f'invalid argument: in {el.tag!r}, {arg=}')


def _add_text(el, s):
    if not s:
        return
    if len(el) == 0:
        el.text = (el.text or '') + s
    else:
        el[-1].tail = (el[-1].tail or '') + s
