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


#:export
class ModelRule(t.Data):
    """Attribute conversion rule"""

    name: str = ''  #: target attribute name
    value: t.Optional[str]  #: constant value
    source: str = ''  #: source attribute
    title: str = ''  #: target attribute display title
    type: t.AttributeType = 'str'  #: target attribute type
    format: t.FormatStr = ''  #: attribute formatter
    expression: str = ''  #: attribute formatter


class Config(t.Config):
    """Data model."""
    rules: t.List[ModelRule]


#:export
class ModelProps(t.Props):
    rules: t.List[ModelRule]


#:export IModel
class Object(gws.Object, t.IModel):
    def __init__(self):
        super().__init__()
        self.rules: t.List[t.ModelRule] = []

    def configure(self):
        super().configure()
        self.rules = self.var('rules')

    @property
    def props(self):
        return t.ModelProps({
            'rules': self.rules
        })

    def apply_to_dict(self, d: dict) -> t.List[t.Attribute]:
        out = []

        for rule in self.rules:
            # @TODO type conversion
            out.append(t.Attribute({
                'title': rule.get('title') or rule.get('name'),
                'name': rule.get('name') or gws.as_uid(rule.get('title', '')),
                'value': self._apply_rule(rule, d),
                'type': rule.get('type') or t.AttributeType.str
            }))

        return out

    def apply(self, atts: t.List[t.Attribute]) -> t.List[t.Attribute]:
        return self.apply_to_dict({a.name: a.value for a in atts})

    def _apply_rule(self, rule: t.ModelRule, d):
        s = rule.get('value')
        if s is not None:
            return s
        s = rule.get('source')
        if s:
            return d.get(s)
        s = rule.get('format')
        if s:
            if '{' not in s:
                return s
            return gws.tools.misc.format_placeholders(s, d)
        return ''
