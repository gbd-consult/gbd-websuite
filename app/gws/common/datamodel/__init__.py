import datetime

import gws
import gws.tools.misc

import gws.types as t

_py_type_to_attr_type = {
    bool: t.AttributeType.bool,
    bytes: t.AttributeType.bytes,
    datetime.date: t.AttributeType.date,
    datetime.datetime: t.AttributeType.datetime,
    datetime.time: t.AttributeType.time,
    float: t.AttributeType.float,
    int: t.AttributeType.int,
    list: t.AttributeType.list,
    str: t.AttributeType.str,
}


def apply(data_model: t.DataModel, atts: dict) -> t.List[t.Attribute]:
    if not data_model:
        return [t.Attribute({
            'name': name,
            'title': name,
            'value': value,
            'type': _py_type_to_attr_type.get(type(value)) or t.AttributeType.str,
        }) for name, value in atts.items()]

    ls = []

    for ac in data_model.attributes:
        # @TODO type conversion
        ls.append(t.Attribute({
            'title': ac.get('title') or ac.get('name'),
            'name': ac.get('name') or gws.as_uid(ac.get('title', '')),
            'value': _apply_attr(ac, atts),
            'type': ac.get('type') or t.AttributeType.str
        }))

    return ls


def _apply_attr(ac: t.Attribute, atts):
    s = ac.get('value')
    if s:
        if '{' not in s:
            return s
        return gws.tools.misc.format_placeholders(s, atts)

    s = ac.get('source')
    return atts.get(s)
