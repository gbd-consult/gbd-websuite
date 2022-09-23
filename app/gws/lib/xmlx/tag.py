import gws
import gws.types as t

from . import element, error


def tag(names: str, *args, **kwargs) -> gws.IXmlElement:

    first = last = None

    for name in names.split():
        el = element.XElement(name)
        if not first:
            first = last = el
        else:
            last.append(el)
            last = el

    if not first:
        raise error.BuildError('invalid name for tag', names)

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
    except TypeError as exc:
        raise error.BuildError('invalid argument for tag', arg) from exc

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

