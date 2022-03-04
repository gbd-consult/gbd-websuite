import datetime
import re

import gws
import gws.tools.misc

import gws.types as t

_default_editor = {
    t.AttributeType.bool: 'checkbox',
    t.AttributeType.bytes: 'file',
    t.AttributeType.date: 'date',
    t.AttributeType.datetime: 'datetime',
    t.AttributeType.float: 'float',
    t.AttributeType.geometry: '',
    t.AttributeType.int: 'int',
    t.AttributeType.intlist: 'list',
    t.AttributeType.str: 'str',
    t.AttributeType.text: 'text',
    t.AttributeType.time: 'time',
}


class AttributeValidator(t.Data):
    type: str
    message: str
    min: t.Optional[float]
    max: t.Optional[float]
    attribute: t.Optional[str]
    pattern: t.Optional[t.Regex]


#:export
class AttributeValidationFailure(t.Data):
    name: str
    message: str


#:export
class AttributeEditor(t.Data):
    type: str
    accept: t.Optional[str]
    items: t.Optional[t.Any]
    max: t.Optional[float]
    min: t.Optional[float]
    multiple: t.Optional[bool]
    pattern: t.Optional[t.Regex]


#:export
class ModelRule(t.Data):
    """Attribute conversion rule"""

    editable: bool = True  #: attribute is editable
    editor: t.Optional[AttributeEditor]
    validators: t.Optional[t.List[AttributeValidator]]
    expression: str = ''  #: attribute conversion expression
    format: t.FormatStr = ''  #: attribute formatter
    name: str = ''  #: target attribute name
    source: str = ''  #: source attribute name
    title: str = ''  #: target attribute title
    type: t.AttributeType = 'str'  #: target attribute type
    value: t.Optional[str]  #: constant value


class Config(t.Config):
    """Data model"""

    crs: t.Optional[t.Crs]  #: CRS for this model
    geometryType: t.Optional[t.GeometryType]  #: specific geometry type
    rules: t.List[ModelRule]  #: attribute conversion rules


class ModelRuleProps(t.Props):
    editable: bool
    editor: AttributeEditor
    name: str
    title: str
    type: str


class ModelProps(t.Props):
    geometryType: str
    crs: str
    rules: t.List[ModelRuleProps]


#:export IModel
class Object(gws.Object, t.IModel):
    def configure(self):
        super().configure()

        p = self.var('rules', default=[])
        self.rules: t.List[t.ModelRule] = [self._configure_rule(r) for r in p]
        self.geometry_type: t.GeometryType = self.var('geometryType')
        self.geometry_crs: t.Crs = self.var('crs')

    @property
    def props(self):
        return ModelProps(
            rules=[
                ModelRuleProps(
                    name=r.name,
                    editable=r.editable,
                    editor=r.editor,
                    title=r.title,
                    type=r.type,
                ) for r in self.rules
            ],
            geometryType=self.geometry_type,
            crs=self.geometry_crs,
        )

    def apply(self, attributes: t.List[t.Attribute]) -> t.List[t.Attribute]:
        attr_values = {a.name: a.value for a in attributes}
        return self.apply_to_dict(attr_values)

    def apply_to_dict(self, attr_values: dict) -> t.List[t.Attribute]:
        return [t.Attribute(
            title=r.title,
            name=r.name,
            value=self._apply_rule(r, attr_values),
            type=r.type,
            editable=r.editable,
        ) for r in self.rules]

    def validate(self, attributes: t.List[t.Attribute]) -> t.List[AttributeValidationFailure]:
        attr_values = {a.name: a.value for a in attributes}
        errors = []
        for r in self.rules:
            err = self._validate_rule(r, attr_values)
            if err:
                errors.append(err)
        return errors

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

    def _apply_rule(self, rule: t.ModelRule, attr_values: dict):
        s = rule.get('value')
        if s is not None:
            return s
        s = rule.get('source')
        if s:
            # @TODO
            if rule.get('type') == 'bytes':
                return None
            return attr_values.get(s)
        s = rule.get('format')
        if s:
            if '{' not in s:
                return s
            return gws.tools.misc.format_placeholders(s, attr_values)
        # no value/source/format present - return values[name]
        return attr_values.get(rule.name)

    def _validate_rule(self, rule: t.ModelRule, attr_values: dict) -> t.Optional[AttributeValidationFailure]:
        v = attr_values.get(rule.name)
        for val in rule.validators:
            if not self._validate(val, v, attr_values):
                return AttributeValidationFailure(name=rule.name, message=val.message)

    def _validate(self, validator: AttributeValidator, value, attr_values) -> bool:
        if validator.type == 'required':
            return not gws.is_empty(value)

        if validator.type == 'length':
            s = gws.as_str(value).strip()
            return validator.min <= len(s) <= validator.max

        if validator.type == 'regex':
            s = gws.as_str(value).strip()
            return bool(re.search(validator.pattern, s))

        if validator.type == 'greaterThan':
            other = attr_values.get(validator.attribute)
            try:
                return value > other
            except TypeError:
                return False

        if validator.type == 'greaterThanOrEqual':
            other = attr_values.get(validator.attribute)
            try:
                return value >= other
            except TypeError:
                return False

        if validator.type == 'lessThan':
            other = attr_values.get(validator.attribute)
            try:
                return value < other
            except TypeError:
                return False

        if validator.type == 'lessThanOrEqual':
            other = attr_values.get(validator.attribute)
            try:
                return value <= other
            except TypeError:
                return False

    def _configure_rule(self, r):
        rule = t.ModelRule(r)

        name = rule.get('name')
        title = rule.get('title')

        if not name:
            name = gws.as_uid(title) if title else rule.get('source')

        if not name:
            raise gws.Error('missing attribute name')

        rule.name = name
        rule.title = title or name

        rule.type = rule.get('type') or t.AttributeType.str

        if not rule.editor:
            rule.editor = AttributeEditor(type=_default_editor.get(rule.type, 'str'))

        rule.validators = r.get('validators') or []

        return rule
