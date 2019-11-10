import datetime

import gws
import gws.tools.misc

import gws.types as t

_py_type_to_attr_type = {
    bool: t.AttributeType.bool,
    bytes: t.AttributeType.bytes,
    datetime.date: t.AttributeType.date,
    datetime.datetime: t.AttributeType.datetime,
    float: t.AttributeType.float,
    int: t.AttributeType.int,
    list: t.AttributeType.list,
    str: t.AttributeType.str,
    datetime.time: t.AttributeType.time,
}


def apply(data_model: t.List[t.AttributeConfig], atts: t.Dict) -> t.Dict[str, t.Attribute]:
    if not data_model:
        return {name: t.Attribute({
            'name': name,
            'title': name,
            'value': value,
            'type': _py_type_to_attr_type.get(type(value)) or t.AttributeType.str,
        }) for name, value in atts.items()}

    ls = {}

    for ac in data_model:
        # @TODO type conversion
        a = t.Attribute({
            'title': ac.get('title') or ac.get('name'),
            'name': ac.get('name') or gws.as_uid(ac.get('title', '')),
            'value': _apply_attr(ac, atts),
            'type': ac.get('type') or t.AttributeType.str
        })
        ls[a.name] = a

    return ls


def _apply_attr(ac: t.AttributeConfig, atts):
    s = ac.get('value')
    if s:
        if '{' not in s:
            return s
        return gws.tools.misc.format_placeholders(s, atts)

    s = ac.get('source')
    return atts.get(s)
