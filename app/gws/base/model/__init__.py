import re

import gws
import gws.types as t
import gws.lib.misc

_DEFAULT_EDITOR = {
    gws.AttributeType.bool: 'checkbox',
    gws.AttributeType.bytes: 'file',
    gws.AttributeType.date: 'date',
    gws.AttributeType.datetime: 'datetime',
    gws.AttributeType.float: 'float',
    gws.AttributeType.geometry: '',
    gws.AttributeType.int: 'int',
    gws.AttributeType.intlist: 'list',
    gws.AttributeType.str: 'str',
    gws.AttributeType.text: 'text',
    gws.AttributeType.time: 'time',
}

_XML_SCHEMA_TYPES = {
    gws.AttributeType.bool: 'xsd:boolean',
    gws.AttributeType.bytes: None,
    gws.AttributeType.date: 'xsd:date',
    gws.AttributeType.datetime: 'datetime',
    gws.AttributeType.float: 'xsd:decimal',
    gws.AttributeType.geometry: None,
    gws.AttributeType.int: 'xsd:integer',
    gws.AttributeType.str: 'xsd:string',
    gws.AttributeType.text: 'xsd:string',
    gws.AttributeType.time: 'xsd:time',
    gws.GeometryType.curve: 'gml:CurvePropertyType',
    gws.GeometryType.geomcollection: 'gml:MultiGeometryPropertyType',
    gws.GeometryType.geometry: 'gml:MultiGeometryPropertyType',
    gws.GeometryType.linestring: 'gml:CurvePropertyType',
    gws.GeometryType.multicurve: 'gml:MultiCurvePropertyType',
    gws.GeometryType.multilinestring: 'gml:MultiCurvePropertyType',
    gws.GeometryType.multipoint: 'gml:MultiPointPropertyType',
    gws.GeometryType.multipolygon: 'gml:MultiSurfacePropertyType',
    gws.GeometryType.multisurface: 'gml:MultiGeometryPropertyType',
    gws.GeometryType.point: 'gml:PointPropertyType',
    gws.GeometryType.polygon: 'gml:PolygonPropertyType',
    gws.GeometryType.polyhedralsurface: 'gml:SurfacePropertyType',
    gws.GeometryType.surface: 'gml:SurfacePropertyType',
}


class AttributeValidator(gws.Data):
    type: str
    message: str
    min: t.Optional[float]
    max: t.Optional[float]
    attribute: t.Optional[str]
    pattern: t.Optional[gws.Regex]


class AttributeValidationFailure(gws.Data):
    name: str
    message: str


class AttributeEditor(gws.Data):
    type: str
    accept: t.Optional[str]
    items: t.Optional[t.Any]
    max: t.Optional[float]
    min: t.Optional[float]
    multiple: t.Optional[bool]
    pattern: t.Optional[gws.Regex]


class Rule(gws.Data):
    """Attribute conversion rule"""

    editable: bool = True  #: attribute is editable
    editor: t.Optional[AttributeEditor]
    validators: t.Optional[t.List[AttributeValidator]]
    expression: str = ''  #: attribute conversion expression
    format: gws.FormatStr = ''  #: attribute formatter
    name: str = ''  #: target attribute name
    source: str = ''  #: source attribute name
    title: str = ''  #: target attribute title
    type: gws.AttributeType = gws.AttributeType.str  #: target attribute type
    value: t.Optional[str]  #: constant value


class Config(gws.Config):
    """Data model"""

    crs: t.Optional[gws.Crs]  #: CRS for this model
    geometryType: t.Optional[gws.GeometryType]  #: specific geometry type
    rules: t.List[Rule]  #: attribute conversion rules


class RuleProps(gws.Props):
    editable: bool
    editor: AttributeEditor
    name: str
    title: str
    type: str


class Props(gws.Props):
    geometryType: str
    crs: str
    rules: t.List[RuleProps]


class Object(gws.Object, gws.IDataModel):
    rules: t.List[Rule]
    geometry_type: gws.GeometryType
    geometry_crs: gws.Crs

    @property
    def props(self):
        return gws.Props(
            rules=[
                gws.Props(
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

    @property
    def attribute_names(self) -> t.List[str]:
        """t.List of attributes used by the model."""
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

    def configure(self):
        self.rules = [self._configure_rule(r) for r in self.var('rules', default=[])]
        self.geometry_type = self.var('geometryType')
        self.geometry_crs = self.var('crs')

    def xml_schema(self, geometry_name='geometry') -> dict:
        schema = {}
        for rule in self.rules:
            typ = _XML_SCHEMA_TYPES.get(rule.type)
            if typ:
                schema[rule.name] = typ

        typ = _XML_SCHEMA_TYPES.get(self.geometry_type)
        if typ:
            schema[geometry_name] = typ

        return schema

    def apply(self, attributes):
        attr_values = {a.name: a.value for a in attributes}
        return self.apply_to_dict(attr_values)

    def apply_to_dict(self, attr_values):
        return [gws.Attribute(
            title=r.title,
            name=r.name,
            value=self._apply_rule(r, attr_values),
            type=r.type,
            editable=r.editable,
        ) for r in self.rules]

    def validate(self, attributes: t.List[gws.Attribute]) -> t.List[AttributeValidationFailure]:
        attr_values = {a.name: a.value for a in attributes}
        errors = []
        for r in self.rules:
            err = self._validate_rule(r, attr_values)
            if err:
                errors.append(err)
        return errors

    def _apply_rule(self, rule: Rule, attr_values: dict):
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
            return gws.lib.misc.format_placeholders(s, attr_values)
        # no value/source/format present - return values[name]
        return attr_values.get(rule.name, '')

    def _validate_rule(self, rule: Rule, attr_values: dict) -> t.Optional[AttributeValidationFailure]:
        v = attr_values.get(rule.name)
        if rule.validators:
            for val in rule.validators:
                if not self._validate(val, v, attr_values):
                    return AttributeValidationFailure(name=rule.name, message=val.message)
        return None

    def _validate(self, validator: AttributeValidator, value, attr_values) -> bool:
        if validator.type == 'required':
            return not gws.is_empty(value)

        if validator.type == 'length':
            s = gws.as_str(value).strip()
            return t.cast(float, validator.min) <= len(s) <= t.cast(float, validator.max)

        if validator.type == 'regex':
            s = gws.as_str(value).strip()
            return bool(re.search(t.cast(str, validator.pattern), s))

        if validator.type == 'greaterThan':
            other = attr_values.get(t.cast(str, validator.attribute))
            try:
                return value > other
            except TypeError:
                return False

        if validator.type == 'lessThan':
            other = attr_values.get(t.cast(str, validator.attribute))
            try:
                return value < other
            except TypeError:
                return False

    def _configure_rule(self, r):
        rule = Rule(r)

        name = rule.get('name')
        title = rule.get('title')

        if not name:
            name = gws.as_uid(title) if title else rule.get('source')

        if not name:
            raise gws.Error('missing attribute name')

        rule.name = name
        rule.title = title or name

        rule.type = rule.get('type') or gws.AttributeType.str

        if not rule.editor:
            rule.editor = AttributeEditor(type=_DEFAULT_EDITOR.get(rule.type, 'str'))

        rule.validators = r.get('validators') or []

        return rule
