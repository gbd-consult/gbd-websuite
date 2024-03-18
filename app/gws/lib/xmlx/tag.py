"""XML builder.

This module provides a single function ``tag``, which creates an Xml Element from a list of arguments.

The first argument to this function is interpreted as a tag name
or a space separated list of tag names, in which case nested elements are created.

The remaining ``*args`` are interpreted as follows:

- a simple string or number value - appended to the text content of the Element
- an Xml Element - appended as a child to the Element
- a dict - attributes of the Element are updated from this dict
- a list, tuple or a generator - used as arguments to ``tag`` to create a child tag

If keyword arguments are given, they are added to the Element's attributes.

**Example:** ::


    tag(
        'geometry gml:Point',
        {'gml:id': 'xy'},
        ['gml:coordinates', '12.345,56.789'],
        srsName=3857
    )

creates the following element: ::

    <geometry>
        <gml:Point gml:id="xy" srsName="3857">
            <gml:coordinates>12.345,56.789</gml:coordinates>
        </gml:Point>
    </geometry>

"""

import gws
import gws.types as t

from . import element, error


def tag(names: str, *args, **kwargs) -> gws.IXmlElement:
    """Build an XML element from arguments.

    Args:
        names: A tag name or names.
        *args: A collection of args.
        **kwargs: Additional attributes.

    Returns:
        An XML element.
    """

    first = last = None

    for name in names.split(' '):
        el = element.XElement(name.strip())
        if not first:
            first = last = el
        else:
            last.append(el)
            last = el

    if not first:
        raise error.BuildError(f'invalid tag name: {names!r}')

    for arg in args:
        _add(last, arg)

    if kwargs:
        _add(last, kwargs)

    return first


def _add(el: gws.IXmlElement, arg):
    if arg is None:
        return

    if isinstance(arg, element.XElement):
        el.append(arg)
        return

    if isinstance(arg, str):
        if arg:
            _add_text(el, arg)
        return

    if isinstance(arg, (int, float, bool)):
        _add_text(el, str(arg).lower())
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
