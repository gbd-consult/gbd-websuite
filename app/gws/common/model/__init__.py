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


class Object(gws.Object, t.ModelObject):
    def configure(self):
        super().configure()
        self.rules = self.var('rules')

    @property
    def props(self):
        return t.ModelProps({
            'rules': self.rules
        })

    def apply(self, atts: t.List[t.Attribute]) -> t.List[t.Attribute]:

        out = []
        att_map = {a.name: a.value for a in atts}

        for rule in self.rules:
            # @TODO type conversion
            out.append(t.Attribute({
                'title': rule.get('title') or rule.get('name'),
                'name': rule.get('name') or gws.as_uid(rule.get('title', '')),
                'value': self._apply_rule(rule, att_map),
                'type': rule.get('type') or t.AttributeType.str
            }))

        return out

    def _apply_rule(self, rule: t.ModelRule, att_map):
        s = rule.get('value')
        if s is not None:
            return s
        s = rule.get('source')
        if s:
            return att_map.get(s)
        s = rule.get('format')
        if s:
            if '{' not in s:
                return s
            return gws.tools.misc.format_placeholders(s, att_map)
        return ''
