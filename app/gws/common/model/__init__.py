import datetime
import re

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

    editable: bool = True  #: attribute is editable
    expression: str = ''  #: attribute conversion expression
    format: t.FormatStr = ''  #: attribute formatter
    name: str = ''  #: target attribute name
    source: str = ''  #: source attribute
    title: str = ''  #: target attribute title
    type: t.AttributeType = 'str'  #: target attribute type
    value: t.Optional[str]  #: constant value


class Config(t.Config):
    """Data model."""

    rules: t.List[ModelRule]
    geometryType: t.Optional[t.GeometryType]
    crs: t.Optional[t.Crs]


#:export
class ModelProps(t.Props):
    rules: t.List[ModelRule]


#:export IModel
class Object(gws.Object, t.IModel):
    def __init__(self):
        super().__init__()
        self.rules: t.List[t.ModelRule] = []
        self.geometry_type: t.GeometryType = ''
        self.geometry_crs: str = ''
        self.is_identity = False

    def configure(self):
        super().configure()

        p = self.var('rules')
        if p:
            self.rules = [self._normalize_rule(r) for r in p]
        self.geometry_type = self.var('geometryType')
        self.geometry_crs = self.var('crs')

    @property
    def props(self):
        return t.ModelProps(
            rules=self.rules
        )

    def apply(self, atts: t.List[t.Attribute]) -> t.List[t.Attribute]:
        return self.apply_to_dict({a.name: a.value for a in atts})

    def apply_to_dict(self, d: dict) -> t.List[t.Attribute]:
        return [t.Attribute(
            title=r.title,
            name=r.name,
            value=self._apply_rule(r, d),
            type=r.type,
            editable=r.editable,
        ) for r in self.rules]

    @property
    def attribute_names(self) -> t.List[str]:
        """List of attributes used by the model."""
        names = set()
        for r in self.rules:
            if r.get('value'):
                continue
            if r.get('source'):
                names.add(r.source)
                continue
            if r.get('format'):
                names.update(re.findall(r'{([\w.]+)', r.format))
                continue
            names.add(r.name)
        return sorted(names)

    def _apply_rule(self, rule: t.ModelRule, d: dict):
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
        # no value/source/format present - return values[name]
        return d.get(rule.name, '')

    def _normalize_rule(self, r):
        if not r.get('title'):
            r.title = r.get('name')
        if not r.get('name'):
            r.name = gws.as_uid(r.get('title'))
        if not r.get('type'):
            r.type = t.AttributeType.str
        return r
