import re
import datetime
import uuid

import sqlalchemy as sa
import sqlalchemy.orm
import geoalchemy2 as geosa

import gws
import gws.config
import gws.gis.shape
import gws.gis.feature
import gws.common.db
import gws.tools.os2
import gws.tools.mime

import gws.types as t


class Error(gws.Error):
    pass


_SCALAR_TYPES = {
    'boolean': sa.Boolean,
    'date': sa.Date,
    'datetime': sa.DateTime,
    'float': sa.Float,
    'integer': sa.Integer,
    'string': sa.String,
    'time': sa.Time,
}


##


class WidgetConfig(t.WithType):
    items: t.Optional[t.List[t.Any]]
    fileField: t.Optional['FieldNameTypeConfig']
    search: t.Optional[str]

class WidgetProps(t.Props):
    type: str
    options: dict


#:export IModelWidget
class Widget(gws.Object, t.IModelWidget):

    def configure(self):
        super().configure()
        self.type = self.var('type')

        self.options = {}
        for k, v in vars(self.config).items():
            if k not in {'type', 'uid'}:
                self.options[k] = v

        p = self.var('items')
        if p:
            self.options['items'] = [
                {
                    'value': it.get('value'),
                    'text': it.get('text') or str(it.get('value')),
                }
                for it in p
            ]

        p = self.var('fileField')
        if p:
            self.options['fileFieldName'] = gws.get(p, 'name')

    @property
    def props(self):
        return WidgetProps(
            type=self.type,
            options=self.options,
        )


class InputWidget(Widget):
    pass


class FeatureListWidget(Widget):
    pass


class DocumentListWidget(Widget):
    pass


class FeatureSelectWidget(Widget):
    pass

class FeatureSuggestWidget(Widget):
    pass


class FileWidget(Widget):
    pass


class DateWidget(Widget):
    pass


class TextareaWidget(Widget):
    pass


class CheckboxWidget(Widget):
    pass


class SelectWidget(Widget):
    pass


class ComboWidget(Widget):
    pass


class MeasurementWidget(Widget):
    pass


class IntegerWidget(Widget):
    pass


class FloatWidget(Widget):
    pass


class GeometryWidget(Widget):
    pass


class ReadonlyWidget(Widget):
    pass


##


class ValidatorConfig(t.WithType):
    pass


#:export IModelValidator
class Validator(gws.Object, t.IModelValidator):
    def configure(self):
        super().configure()

        self.type = self.var('type')

    def validate_value(self, field, value, attributes):
        return None, value


##


def _expr_now():
    return datetime.datetime.now().strftime('%Y-%m-%d')


def _expr_uuid():
    return uuid.uuid4()


EXPRESSIONS = {
    'now()': _expr_now,
    'uuid_generate_v4()': _expr_uuid,
}


##

class ValidationError(Exception):
    pass


class FieldValueConfig(t.WithType):
    value: t.Optional[t.Any]
    expression: t.Optional[str]


class FieldRelationConfig(t.Config):
    modelUid: str
    fieldName: str
    title: str
    discriminator: str


class FieldNameTypeConfig:
    type: t.Optional[str]
    name: str


class FieldLinkConfig:
    tableName: str
    keyName: str


class FieldConfig(t.WithType):
    name: str
    title: t.Optional[str]

    widget: t.Optional[WidgetConfig]

    relation: t.Optional[FieldRelationConfig]
    relations: t.Optional[t.List[FieldRelationConfig]]

    foreignKey: t.Optional[FieldNameTypeConfig]
    discriminatorKey: t.Optional[FieldNameTypeConfig]
    link: t.Optional[FieldLinkConfig]

    basePath: t.Optional[str]

    default: t.Optional[FieldValueConfig]
    validators: t.Optional[t.List[ValidatorConfig]]

    minValue: t.Optional[float]
    maxValue: t.Optional[float]

    minLen: t.Optional[int]
    maxLen: t.Optional[int]
    pattern: t.Optional[str]

    geometryType: t.Optional[str]

    isRequired: bool = False
    isEditable: bool = True
    isPrimaryKey: bool = False
    isUnique: bool = False
    isSearchable: bool = False
    isViewable: bool = True


class FieldRelationProps(t.Props):
    type: str
    modelUid: str
    fieldName: str
    title: str


class FieldProps(t.Props):
    type: str
    dataType: t.Optional[str]
    geometryType: t.Optional[str]
    name: str
    title: str
    relations: t.Optional[t.List[FieldRelationProps]]
    widget: t.Optional[WidgetProps]


#:export IModelField
class Field(gws.Object, t.IModelField):
    type: str
    name: str
    model: t.IModel
    widget: t.Optional[t.IModelWidget]
    validators: t.List[t.IModelValidator]
    data_type: str
    geometry_type: str

    def configure(self):
        super().configure()

        self.type = self.var('type')
        self.name = self.var('name')
        self.title = self.var('title', default=self.name)
        self.geometry_type = self.var('geometryType', default='').upper()

        self.default = self.var('default')

        self.is_primary_key = self.var('isPrimaryKey')
        self.is_editable = self.var('isEditable')
        self.is_required = self.var('isRequired')
        self.is_searchable = self.var('isSearchable')
        self.is_viewable = self.var('isViewable')

        self.min_value = self.var('minValue')
        self.max_value = self.var('maxValue')
        self.min_len = self.var('minLen')
        self.max_len = self.var('maxLen')
        self.pattern = self.var('pattern')

        self.widget = None
        p = self.var('widget')
        if p:
            cls = _WIDGET_TYPES.get(p.type)
            if not cls:
                raise Error(f'no widget {p.type}')
            self.widget = t.cast(t.IModelWidget, self.create_child(cls, p))

        self.validators = []
        ps = self.var('validators')
        if ps:
            for p in ps:
                cls = _VALIDATOR_TYPES.get(p.type)
                if not cls:
                    raise Error(f'no validator {p.type}')
                v = t.cast(t.IModelValidator, self.create_child(cls, p))
                v.field = self
                self.validators.append(v)

    def sa_columns(self, columns):
        pass

    def sa_properties(self, properties):
        pass

    def sa_adapt_select(self, state):
        pass

    def get_default(self):
        if not self.default:
            return
        v = self.default.get('value')
        if v is not None:
            return v
        v = self.default.get('expression')
        if v is not None:
            return EXPRESSIONS[v]()

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, depth: int):
        pass

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, depth: int):
        pass

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        pass

    def write_to_orm(self, fe: t.IFeature, obj):
        pass

    def validate(self, fe: t.IFeature, errors):
        value = fe.attr(self.name)

        if value is None:
            if self.is_required:
                errors.append(t.FeatureError(name=self.name, error='validationErrorNull'))
            return

        if not self.validators:
            try:
                fe.attributes[self.name] = self.validate_value(value)
            except ValidationError as err:
                errors.append(t.FeatureError(name=self.name, error=err.args[0]))
                return

        # for v in self.validators:
        #     err = v.validate_value(self, value, attributes)
        #     if err:
        #         errors.append(t.FeatureError(name=self.name, error=err))

    def validate_value(self, value):
        return value

    @property
    def props(self):
        p = t.Props(
            type=self.type,
            name=self.name,
            title=self.title,
            dataType=self.data_type,
        )

        if self.widget:
            p.widget = self.widget.props

        if hasattr(self, 'geometry_type'):
            p.geometryType = getattr(self, 'geometry_type', None)

        return p


class ScalarField(Field):
    def sa_columns(self, columns):
        col = sa.Column(self.name, _SCALAR_TYPES[self.data_type], primary_key=self.is_primary_key)
        columns.append(col)

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = props.attributes.get(self.name)
        if val is not None:
            fe.attributes[self.name] = val
            return
        val = self.get_default()
        if val is not None:
            fe.attributes[self.name] = val
            return

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val
            return
        val = self.get_default()
        if val is not None:
            props.attributes[self.name] = val
            return

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        val = getattr(obj, self.name, None)
        if val is not None:
            fe.attributes[self.name] = val
            return
        val = self.get_default()
        if val is not None:
            fe.attributes[self.name] = val
            return

    def write_to_orm(self, fe: t.IFeature, obj):
        if self.name in fe.attributes:
            setattr(obj, self.name, fe.attributes[self.name])
            return
        val = self.get_default()
        if val is not None:
            setattr(obj, self.name, val)


class StringField(ScalarField):
    data_type = 'string'

    def sa_adapt_select(self, state):
        model = t.cast(t.IDbModel, self.model)

        if state.args.keyword and self.is_searchable:
            state.keyword_conds.append((
                getattr(model.get_class(), self.name).ilike(
                    '%' + _escape_like(state.args.keyword) + '%', escape='\\')))

    def validate_value(self, value):
        v = str(value).strip()
        if v == '':
            if self.is_required:
                raise ValidationError('validationErrorNull')
            return v
        if self.min_len is not None and len(v) < self.min_len:
            raise ValidationError('validationErrorStringTooShort')
        if self.max_len is not None and len(v) > self.max_len:
            raise ValidationError('validationErrorStringTooLong')
        if self.pattern and not re.match(str(self.pattern), v):
            raise ValidationError('validationErrorPattern')
        return v


class IntegerField(ScalarField):
    data_type = 'integer'

    def convert(self, val):
        if isinstance(val, int):
            return val
        v = str(val).strip()
        if not v:
            return None
        return int(v)

    def validate_value(self, value):
        try:
            v = self.convert(value)
        except:
            raise ValidationError('validationErrorInvalidInteger')
        if v is None:
            if self.is_required:
                raise ValidationError('validationErrorNull')
            return v
        if self.min_value is not None and v < self.min_value:
            raise ValidationError('validationErrorNumberTooSmall')
        if self.max_value is not None and v > self.max_value:
            raise ValidationError('validationErrorNumberTooBig')
        return v


class FloatField(ScalarField):
    data_type = 'float'

    def convert(self, val):
        if isinstance(val, (int, float)):
            return float(val)
        v = str(val).strip()
        if not v:
            return None
        return float(v)

    def validate_value(self, value):
        try:
            v = self.convert(value)
        except:
            raise ValidationError('validationErrorInvalidFloat')
        if v is None:
            if self.is_required:
                raise ValidationError('validationErrorNull')
            return v
        if self.min_value is not None and v < self.min_value:
            raise ValidationError('validationErrorNumberTooSmall')
        if self.max_value is not None and v > self.max_value:
            raise ValidationError('validationErrorNumberTooBig')
        return v


class DateField(ScalarField):
    data_type = 'date'

    def convert(self, val):
        if isinstance(val, datetime.date):
            return val
        if isinstance(val, datetime.datetime):
            return datetime.date(val.year, val.month, val.day)
        v = str(val).strip()
        if not v:
            return None
        m = re.match(r'^(\d{4})-(\d\d)-(\d\d)$', v)
        if not m:
            raise ValueError('invalid date')
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def validate_value(self, value):
        try:
            v = self.convert(value)
        except:
            raise ValidationError('validationErrorInvalidDate')
        return v


def _is_record_with_type(val, type):
    return isinstance(val, (dict, t.Data)) and val.get('type') == type


class FileValue(t.Data):
    name: str
    content: t.Optional[bytes]
    mime: t.Optional[str]


class FileField(ScalarField):
    data_type = 'string'

    def normalize_file_name(self, s):
        p = gws.tools.os2.parse_path(s)
        return gws.as_uid(p['name']) + '.' + gws.as_uid(p['extension'])

    def store_file(self, fname, content):
        file_path = self.var('basePath') + '/' + fname
        gws.write_file_b(file_path, content)
        return True

    def get_file_path(self, fe: t.IFeature):
        val = fe.attr(self.name)
        if val:
            return self.var('basePath') + '/' + val.name

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = props.attributes.get(self.name)
        if val is not None:
            fe.attributes[self.name] = FileValue(
                name=gws.get(val, 'name'),
                content=gws.get(val, 'content'))

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = fe.attr(self.name)
        if val:
            props.attributes[self.name] = val

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        fname = getattr(obj, self.name, None)
        if fname is not None:
            fe.attributes[self.name] = FileValue(name=fname, mime=gws.tools.mime.for_path(fname))

    def write_to_orm(self, fe: t.IFeature, obj):
        val = fe.attr(self.name)
        if val is not None:
            fname = self.normalize_file_name(gws.get(val, 'name'))
            content = gws.get(val, 'content')
            if fname and content:
                self.store_file(fname, content)
                setattr(obj, self.name, fname)


class GeometryField(ScalarField):
    data_type = 'string'
    geometry_type = ''

    def sa_adapt_select(self, state):
        if state.args.shape:
            model = t.cast(t.IDbModel, self.model)

            shape = state.args.shape.tolerance_polygon(state.args.map_tolerance)
            shape = shape.transformed_to(model.get_table().geometry_crs)
            state.geometry_conds.append(sa.func.st_intersects(
                getattr(model.get_class(), self.name),
                sa.cast(shape.ewkb_hex, geosa.Geometry())))

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = props.attributes.get(self.name)
        if val is not None:
            fe.attributes[self.name] = gws.gis.shape.from_props(val)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.props

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        val = getattr(obj, self.name, None)
        if val is not None:
            fe.attributes[self.name] = gws.gis.shape.from_wkb_hex(val)

    def write_to_orm(self, fe: t.IFeature, obj):
        val = fe.attributes.get(self.name)
        if val is not None:
            setattr(obj, self.name, val.ewkb_hex)


##


class RelatedField(Field):
    model: t.IDbModel

    def configure(self):
        super().configure()

        self.relations = []

        self.link_table_name = self.var('link.tableName')
        self.link_key_name = self.var('link.keyName')

        self.foreign_key_name = self.var('foreignKey.name')
        self.foreign_key_type = self.var('foreignKey.type')

        self.discriminator_name = self.var('discriminatorKey.name')
        self.discriminator_type = self.var('discriminatorKey.type')

        self.relations = []

        rels = []
        p = self.var('relation')
        if p:
            rels.append(p)
        p = self.var('relations')
        if p:
            rels.extend(p)

        for r in rels:
            self.relations.append(t.Data(
                model_uid=r.modelUid,
                field_name=r.fieldName,
                title=r.title,
                discriminator=r.discriminator,
            ))

    @property
    def props(self):
        p = super().props
        p.relations = []
        for r in self.relations:
            p.relations.append(t.Props(
                modelUid=r.model_uid,
                fieldName=r.field_name,
                title=r.title,
            ))
        return p

    def sa_adapt_select_selectin(self, state):
        if state.args.depth and state.args.depth > 0:
            state.sel = state.sel.options(
                sa.orm.selectinload(
                    getattr(self.model.get_class(), self.name)
                ))

    def first_related_model(self):
        return registry().get_model(self.relations[0].model_uid)

    def relation_for_model(self, model):
        for r in self.relations:
            if r.model_uid == model.uid:
                return r

    ##


##


class RelatedFeatureField(RelatedField):
    data_type = 'feature'

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        if depth <= 0:
            return

        val = props.attributes.get(self.name)
        if val is None:
            return

        rel_model = self.first_related_model()
        uid = val.get('attributes', {}).get(rel_model.key_name)
        fe.attributes[self.name] = rel_model.get_feature(uid, depth - 1)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        fe2 = fe.attributes.get(self.name)
        if fe2:
            props.attributes[self.name] = fe2.model.feature_props(fe2, depth - 1)

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        if depth <= 0:
            return
        rel_model = self.first_related_model()
        obj2 = getattr(obj, self.name, None)
        if obj2:
            # exclude = (exclude or []) + [self.relations[0].field_name]
            fe.attributes[self.name] = rel_model.feature_from_orm(obj2, depth - 1)

    def write_to_orm(self, fe: t.IFeature, obj):
        rel_model = self.first_related_model()
        fe2 = fe.attributes.get(self.name)
        if fe2:
            setattr(obj, self.name, rel_model.get_object(fe2.uid))


class RelatedFeatureListField(RelatedField):
    data_type = 'featureList'

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        if depth <= 0:
            return
        val = props.attributes.get(self.name)
        if val is None:
            return

        rel_model = self.first_related_model()
        uids = [f.get('attributes').get(rel_model.key_name) for f in val]
        fe.attributes[self.name] = [rel_model.get_feature(uid, depth - 1) for uid in uids]

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        flist = fe.attributes.get(self.name)
        if flist:
            props.attributes[self.name] = [fe2.model.feature_props(fe2, depth - 1) for fe2 in flist]

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        if depth <= 0:
            return

        rel_model = self.first_related_model()

        objlist = getattr(obj, self.name, None)
        if objlist:
            # exclude = (exclude or []) + [self.relations[0].field_name]
            fe.attributes[self.name] = [
                rel_model.feature_from_orm(o, depth - 1)
                for o in objlist
            ]

    def write_to_orm(self, fe: t.IFeature, obj):
        rel_model = self.first_related_model()
        flist = fe.attributes.get(self.name)
        if flist:
            setattr(obj, self.name, [rel_model.get_object(f.uid) for f in flist])


"""
relatedFeature   M->1

    HOUSE -> street_id -> STREET

    HOUSE.street:      
        type = STREET
        relatedModel = STREET (parent)
        relatedField (optional) = 1->M field in parent = STREET.houses
        foreignKey = foreign key name (street_id)
"""


class Field_relatedFeature(RelatedFeatureField):
    sa_adapt_select = RelatedFeatureField.sa_adapt_select_selectin

    def sa_columns(self, columns):
        rel_model = self.first_related_model()
        rel_keys = rel_model.get_keys()
        columns.append(sa.Column(self.foreign_key_name, sa.ForeignKey(rel_keys[0])))

    def sa_properties(self, properties):
        rel_model = self.first_related_model()
        rel_cls = rel_model.get_class()
        kwargs = {}
        if self.relations[0].field_name:
            kwargs['back_populates'] = self.relations[0].field_name
        properties[self.name] = sa.orm.relationship(rel_cls, **kwargs)


"""
relatedFeatureList   1->M
    
    STREET -> HOUSE (list)

    STREET.houses:
        type = Array<HOUSE>
        relatedModel = HOUSE (child)
        relatedField (mandatory) = HOUSE.street (M->1 field in child)

"""


class Field_relatedFeatureList(RelatedFeatureListField):
    sa_adapt_select = RelatedFeatureListField.sa_adapt_select_selectin

    def sa_properties(self, properties):
        rel_model = self.first_related_model()
        rel_cls = rel_model.get_class()
        kwargs = {}
        kwargs['back_populates'] = self.relations[0].field_name
        properties[self.name] = sa.orm.relationship(rel_cls, **kwargs)


"""
relatedMultiFeatureList   1->M 

    STREET.id = HOUSE.street_id
    STREET.id = TREE.street_id
    
    STREET.objects
        type = Array<HOUSE|TREE>
        relations = [
            { modelUid: HOUSE title 'House' fieldName: street } 
            { modelUid: TREE  title 'Tree' fieldName: street } 
        ]
        
    symmetric = parentFeature
"""


class Field_relatedMultiFeatureList(RelatedFeatureListField):

    def sa_properties(self, properties):
        for r in self.relations:
            rel_model = registry().get_model(r.model_uid)
            rel_cls = rel_model.get_class()
            kwargs = {}
            kwargs['back_populates'] = r.field_name
            key = self.name + ':' + r.model_uid
            properties[key] = sa.orm.relationship(rel_cls, **kwargs)

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        if depth <= 0:
            return

        flist = []

        for r in self.relations:
            key = self.name + ':' + r.model_uid
            val = getattr(obj, key, None)
            if val is None:
                continue
            rel_model = registry().get_model(r.model_uid)
            for obj2 in val:
                flist.append(rel_model.feature_from_orm(obj2, depth - 1))

        if flist:
            fe.attributes[self.name] = flist

    def write_to_orm(self, fe: t.IFeature, obj):
        val = fe.attributes.get(self.name)
        if val is None:
            return

        d = {}

        for fe2 in val:
            r = self.relation_for_model(fe2.model)
            if r:
                key = self.name + ':' + r.model_uid
                if key not in d:
                    d[key] = []
                d[key].append(fe2.model.get_object(fe2.uid))

        for key, fs in d.items():
            setattr(obj, key, fs)


"""

mmLinkRelation  STREET <-- <LINK> --> DISTRICT
    
    STREET.districts
        type = Array<DISTRICT>
        relatedModel = DISTRICT
        relatedField (mandatory) = DISTRICT.streets
        linkTableName = LINK
        linkKeyName = LINK.street_id (=this pk)

    DISTRICT.streets:
        type = Array<STREET>
        relatedModel = STREET
        relatedField (mandatory) = STREET.districts
        linkTableName = LINK
        linkKeyName = LINK.district_id (=this pk)

    LINK
        street_id = STREET.id
        district_id = DISTRICT.id
"""


class Field_relatedLinkedFeatureList(RelatedFeatureListField):
    sa_adapt_select = RelatedFeatureListField.sa_adapt_select_selectin

    def sa_properties(self, properties):
        rel_model = self.first_related_model()

        own_keys = self.model.get_keys()
        rel_keys = rel_model.get_keys()

        own_link_key = self.link_key_name
        rel_link_key = rel_model.get_field(self.relations[0].field_name).link_key_name

        cols = [
            sa.Column(own_link_key, sa.ForeignKey(own_keys[0]), primary_key=True),
            sa.Column(rel_link_key, sa.ForeignKey(rel_keys[0]), primary_key=True),
        ]

        link_table = registry().create_table(self.link_table_name, cols)

        kwargs = {}
        kwargs['secondary'] = link_table
        kwargs['back_populates'] = self.relations[0].field_name

        rel_cls = rel_model.get_class()
        properties[self.name] = sa.orm.relationship(rel_cls, **kwargs)


"""
relatedGenericFeatureList   1->M 
    
    https://docs.sqlalchemy.org/en/14/_modules/examples/generic_associations/generic_fk.html
    
    STREET.id -> IMAGE.object_id
    HOUSE.id  -> IMAGE.object_id

    STREET.images
        type = Array<IMAGE>
        relatedModel = IMAGE
        relatedField (mandatory) = IMAGE.object

    HOUSE.images:
        type = Array<IMAGE>
        relatedModel = IMAGE
        relatedField (mandatory) = IMAGE.object
"""


class Field_relatedGenericFeatureList(RelatedFeatureListField):

    def sa_properties(self, properties):
        rel_model = self.first_related_model()

        own_cls = self.model.get_class()
        rel_cls = rel_model.get_class()

        own_pk = self.model.get_keys()
        rel_fk_name = rel_model.get_field(self.relations[0].field_name).foreign_key_name

        kwargs = {}
        kwargs['primaryjoin'] = getattr(own_cls, own_pk[0].name) == getattr(rel_cls, rel_fk_name)
        kwargs['foreign_keys'] = getattr(rel_cls, rel_fk_name)

        properties[self.name] = sa.orm.relationship(rel_cls, **kwargs)


"""
relatedGenericFeature   M->1 
    
    https://docs.sqlalchemy.org/en/14/_modules/examples/generic_associations/generic_fk.html
    
    IMAGE.object_id -> STREET.id 
    IMAGE.object_id -> HOUSE.id 

    IMAGE.object
        type = GenericFeature (=only id)
"""


class Field_relatedGenericFeature(RelatedFeatureField):

    def sa_columns(self, columns):
        col = sa.Column(self.foreign_key_name, _SCALAR_TYPES[self.foreign_key_type])
        columns.append(col)

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = props.attributes.get(self.name)
        gws.p('read_from_props val=', val)
        if val is None:
            return

        if isinstance(val, (int, str)):
            uid = val
        elif isinstance(val, t.IFeature):
            uid = val.uid
        else:
            key = gws.get(val, 'keyName')
            uid = gws.get(val, ['attributes', key])

        gws.p('read_from_props uid=', uid)
        if uid:
            fe.attributes[self.name] = generic_feature(uid=uid)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.model.feature_props(val, depth - 1)
            return

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        val = getattr(obj, self.foreign_key_name, None)
        if val is not None:
            fe.attributes[self.name] = generic_feature(uid=val)
            return

    def write_to_orm(self, fe: t.IFeature, obj):
        val = fe.attributes.get(self.name)
        if val is not None:
            setattr(obj, self.foreign_key_name, val.uid)
            return


"""
omDiscriminatedRelation   1->M 
    
    https://docs.sqlalchemy.org/en/14/_modules/examples/generic_associations/generic_fk.html
    
    STREET.id -> IMAGE.object_id, IMAGE.<discriminator> = 'street'
    HOUSE.id  -> IMAGE.object_id, IMAGE.<discriminator> = 'house'

    STREET.images
        type = Array<IMAGE>
        relatedModel = IMAGE
        relatedField (mandatory) = IMAGE.object

    HOUSE.images:
        type = Array<IMAGE>
        relatedModel = IMAGE
        relatedField (mandatory) = IMAGE.object
"""


class Field_relatedDiscriminatedFeatureList(RelatedFeatureListField):

    def sa_properties(self, properties):
        rel_model = self.first_related_model()

        own_cls = self.model.get_class()
        rel_cls = rel_model.get_class()

        rel_field = rel_model.get_field(self.relations[0].field_name)

        rel = rel_field.relation_for_model(self.model)
        own_pk = self.model.get_keys()
        rel_fk_name = rel_field.foreign_key_name

        kwargs = {}
        kwargs['primaryjoin'] = sa.and_(
            getattr(own_cls, own_pk[0].name) == getattr(rel_cls, rel_fk_name),
            getattr(rel_cls, rel_field.discriminator_name) == rel.discriminator
        )
        kwargs['foreign_keys'] = getattr(rel_cls, rel_fk_name)

        properties[self.name] = sa.orm.relationship(rel_cls, **kwargs)

    def write_to_orm(self, fe: t.IFeature, obj):
        super().write_to_orm(fe, obj)

        rel_model = self.first_related_model()
        rel_obj_list = getattr(obj, self.name)
        if rel_obj_list:
            rel_field = rel_model.get_field(self.relations[0].field_name)
            rel = rel_field.relation_for_model(self.model)
            for o in rel_obj_list:
                setattr(o, rel_field.discriminator_name, rel.discriminator)


"""
moDiscriminatedRelation   M->1 
    
    https://docs.sqlalchemy.org/en/14/_modules/examples/generic_associations/generic_fk.html
    
    STREET.id -> IMAGE.object_id, IMAGE.discriminator = 'street'
    HOUSE.id  -> IMAGE.object_id, IMAGE.discriminator = 'house'

    IMAGE.object
        type = STREET|HOUSE
        discriminatorName 'object_type'
        discriminatorType 'string'
        relations = [
            { uid: STREET title 'Street' fieldName: objects discriminator: 'street' } 
            { uid: HOUSE  title 'House'  fieldName: objects discriminator: 'house'  } 
        ]
        
"""


class Field_relatedDiscriminatedFeature(RelatedFeatureField):

    def relation_for_discriminator(self, d):
        for r in self.relations:
            if r.discriminator == d:
                gws.log.debug(f'found model {r.model_uid!r} for discriminator={d!r}')
                return r

    def sa_columns(self, columns):
        col = sa.Column(self.foreign_key_name, _SCALAR_TYPES[self.foreign_key_type])
        columns.append(col)
        col = sa.Column(self.discriminator_name, _SCALAR_TYPES[self.discriminator_type])
        columns.append(col)

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        if depth <= 0:
            return
        val = props.attributes.get(self.name)
        if val is not None:
            rel_model = registry().get_model(val.get('modelUid'))
            uid = val.get('attributes', {}).get(rel_model.key_name)
            fe.attributes[self.name] = rel_model.get_feature(uid, depth - 1)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, depth):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.props

    def read_from_orm(self, fe: t.IFeature, obj, depth):
        val = getattr(obj, self.foreign_key_name, None)
        rel = self.relation_for_discriminator(getattr(obj, self.discriminator_name, None))
        if val is not None:
            rel_model = registry().get_model(rel.model_uid)
            sobj = rel_model.get_object(val)
            fe.attributes[self.name] = rel_model.feature_from_orm(sobj)
            return

    def write_to_orm(self, fe: t.IFeature, obj):
        val = fe.attributes.get(self.name)
        if val is not None:
            rel = self.relation_for_model(val.model)
            setattr(obj, self.foreign_key_name, val.uid)
            setattr(obj, self.discriminator_name, rel.discriminator)
            gws.log.debug(f'@@ {self.foreign_key_name}={val.uid!r}, {self.discriminator_name}={rel.discriminator!r}')


##

_WIDGET_TYPES = {
    'combo': ComboWidget,
    'date': DateWidget,
    'file': FileWidget,
    'float': FloatWidget,
    'geometry': GeometryWidget,
    'integer': IntegerWidget,
    'readonly': ReadonlyWidget,
    'featureList': FeatureListWidget,
    'documentList': FeatureListWidget,
    'featureSelect': FeatureSelectWidget,
    'featureSuggest': FeatureSuggestWidget,
    'select': SelectWidget,
    'input': InputWidget,
    'textarea': TextareaWidget,
    'checkbox': CheckboxWidget,
    'measurement': MeasurementWidget,
}

_FIELD_TYPES = {
    'string': StringField,
    'integer': IntegerField,
    'float': FloatField,
    'date': DateField,
    'file': FileField,

    'geometry': GeometryField,

    'relatedFeature': Field_relatedFeature,
    'relatedFeatureList': Field_relatedFeatureList,
    'relatedMultiFeatureList': Field_relatedMultiFeatureList,
    'relatedLinkedFeatureList': Field_relatedLinkedFeatureList,
    'relatedGenericFeatureList': Field_relatedGenericFeatureList,
    'relatedGenericFeature': Field_relatedGenericFeature,
    'relatedDiscriminatedFeatureList': Field_relatedDiscriminatedFeatureList,
    'relatedDiscriminatedFeature': Field_relatedDiscriminatedFeature,
}

_VALIDATOR_TYPES = {
}


##


##


class Config(t.WithAccess):
    """Model configuration"""

    fields: t.List[FieldConfig]
    sqlFilter: t.Optional[str]


class Props(t.Data):
    uid: str
    layerUid: str
    keyName: str
    geometryName: str
    fields: t.List[FieldProps]


class SelectState(t.Data):
    sel: sa.sql.expression.select
    args: t.SelectArgs
    keyword_conds: t.List[t.Any]
    geometry_conds: t.List[t.Any]


#:export IModel
class Object(gws.Object, t.IModel):
    layer: t.Optional[t.ILayer]
    fields: t.List[t.IModelField] = []
    key_name: str = ''
    geometry_name: str = ''
    keyword_columns: t.List[str] = []

    def configure(self):
        super().configure()

        self.layer = None

        self.key_name = ''
        self.geometry_name = ''

        self.fields = []

        for p in self.var('fields'):
            cls = _FIELD_TYPES[p.type]
            if not cls:
                raise Error(f'no field {p.type!r}')

            f = t.cast(t.IModelField, self.create_child(cls, p))
            f.model = self
            self.fields.append(f)

            if isinstance(f, GeometryField) and not self.geometry_name:
                self.geometry_name = f.name
            if f.is_primary_key:
                self.key_name = f.name

    def select(self, args: t.SelectArgs) -> t.List[t.IFeature]:
        return []

    def save(self, fe: t.IFeature) -> t.IFeature:
        return fe

    def delete(self, fe: t.IFeature):
        pass

    def reload(self, fe: t.IFeature, depth: int = 0):
        pass

    def validate(self, fe: t.IFeature) -> t.List[t.FeatureError]:
        return []

    def init_feature(self):
        fe = gws.gis.feature.Feature(self)
        fe.key_name = self.key_name
        fe.geometry_name = self.geometry_name
        if self.layer:
            fe.layer = self.layer
        return fe

    def new_feature(self):
        fe = self.init_feature()
        fe.is_new = True
        return fe

    def get_feature(self, uid, depth=0) -> t.Optional[t.IFeature]:
        return None

    def feature_from_props(self, props: t.FeatureProps, depth=0):
        fe = self.init_feature()
        fe.elements = props.get('elements') or {}
        fe.category = props.get('category') or ''
        fe.is_new = bool(props.get('isNew'))
        for f in self.fields:
            f.read_from_props(fe, props, depth)

        # s = p.get('style')
        # if isinstance(s, dict):
        #     s = t.StyleProps(s)
        # f.style = gws.common.style.from_props(s)

        return fe

    def feature_props(self, fe, depth=0):
        props = t.FeatureProps(
            attributes={},
            keyName=self.key_name,
            geometryName=self.geometry_name,
            modelUid=self.uid,
            layerUid=self.layer.uid if self.layer else None,
            category=fe.category or '',
            elements=fe.elements or {},
            isNew=fe.is_new,
        )

        for f in self.fields:
            f.write_to_props(fe, props, depth)

        if fe.errors:
            props.errors = fe.errors

        return props

    def get_field(self, name: str) -> t.Optional[t.IModelField]:
        for f in self.fields:
            if f.name == name:
                return f

    @property
    def props(self):
        p = Props(uid=self.uid, fields=[f.props for f in self.fields])
        if self.layer:
            p.layerUid = self.layer.uid
        if self.key_name:
            p.keyName = self.key_name
        if self.geometry_name:
            p.geometryName = self.geometry_name
        return p


#:export IDbModel
class DbModel(Object, t.IDbModel):
    def configure(self):
        super().configure()
        self.sql_filter = self.var('sqlFilter')

    def select(self, args: t.SelectArgs) -> t.List[t.IFeature]:

        sel = self.sa_make_select(args)
        if sel is None:
            return []

        cls = self.get_class()
        flist = []

        cursor = self.sa_session().execute(sel)
        for row in cursor.unique().all():
            obj = getattr(row, cls.__name__)
            fe = self.feature_from_orm(obj, depth=args.depth or 0)
            flist.append(fe)

        return flist

    def save(self, fe: t.IFeature):

        if fe.is_new:
            obj = self.get_class()()
        else:
            obj = self.get_object(fe.uid)

        for f in self.fields:
            f.write_to_orm(fe, obj)

        if fe.is_new:
            self.sa_session().add(obj)

        fe._model_data = {'object': obj}
        return fe

    def delete(self, fe: t.IFeature):
        self.sa_session().delete(self.get_object(fe.uid))

    def reload(self, fe: t.IFeature, depth: int = 0):
        md = getattr(fe, '_model_data', None)
        if md:
            f2 = self.feature_from_orm(md['object'], depth=depth)
            fe.attributes = f2.attributes
        return fe

    def validate(self, fe) -> t.List[t.FeatureError]:
        errors = []
        for f in self.fields:
            f.validate(fe, errors)
        return errors

    def get_feature(self, uid, depth=0):
        obj = self.get_object(uid)
        if obj:
            return self.feature_from_orm(obj, depth)

    def feature_from_orm(self, obj, depth=0):
        fe = self.init_feature()

        # gws.log.debug(f"FETCH {self.uid} uid={getattr(obj, self.key_name, '?')}")

        for f in self.fields:
            f.read_from_orm(fe, obj, depth)

        fe._model_data = {'object': obj}
        return fe

    def get_db(self):
        if self.layer:
            return getattr(self.layer, 'db', None)

    def get_table(self) -> t.SqlTable:
        if self.layer:
            return getattr(self.layer, 'table', None)

    def get_class(self):
        return registry().get_class(self.uid)

    def get_keys(self):
        return registry().get_keys(self.uid)

    def get_object(self, uid):
        return self.sa_session().get(self.get_class(), uid)

    def sa_session(self):
        return registry().session(self.get_db())

    def sa_make_select(self, args: t.SelectArgs):
        cls = self.get_class()

        state = SelectState(sel=sa.select(cls), args=args, keyword_conds=[], geometry_conds=[])

        for f in self.fields:
            f.sa_adapt_select(state)

        if args.keyword and not state.keyword_conds:
            return
        if state.keyword_conds:
            state.sel = state.sel.where(sa.or_(*state.keyword_conds))

        if args.shape and not state.geometry_conds:
            return
        if state.geometry_conds:
            state.sel = state.sel.where(sa.or_(*state.geometry_conds))

        if args.uids:
            state.sel = state.sel.where(getattr(cls, self.key_name).in_(args.uids))

        if args.extra_where:
            s = sa.text(args.extra_where[0])
            if len(args.extra_where) > 1:
                s = s.bindparams(**args.extra_where[1])
            state.sel = state.sel.where(s)

        if self.sql_filter:
            state.sel = state.sel.where(sa.text(self.sql_filter))

        gws.log.debug(f'SA_MAKE_SELECT: {str(state.sel)}')

        return state.sel


def _escape_like(s, escape='\\'):
    return (
        s
            .replace(escape, escape + escape)
            .replace('%', escape + '%')
            .replace('_', escape + '_'))


##

class GenericModel(Object):
    key_name = 'uid'
    geometry_name = 'geometry'
    layer = None

    def feature_from_props(self, props: t.FeatureProps, depth=0):
        fe = super().feature_from_props(props, depth)
        fe.attributes = gws.get(props, 'attributes', default={})
        return fe

    def feature_props(self, fe, depth=0):
        props = super().feature_props(fe, depth)
        for k, v in fe.attributes.items():
            if hasattr(v, 'props'):
                v = v.props
            props.attributes[k] = v
        return props


_genericModel = GenericModel()
_genericModel.uid = '_genericModel'


def generic():
    return _genericModel


def generic_feature(**args):
    m = generic()
    fe = m.new_feature()

    for k, v in args.items():
        if k == 'uid':
            fe.attributes[m.key_name] = v
        elif k == 'shape':
            fe.attributes[m.geometry_name] = v
        elif k == 'attributes':
            fe.attributes.update(v)
        elif k == 'category':
            fe.category = v

    return fe


##


class SaBase:
    feature: t.IFeature


class ModelRegistry:
    def __init__(self, root):
        self.root = root
        self.ms = {}
        self.engines: t.Dict[str, sa.engine.Engine] = {}
        self.sessions: t.Dict[str, sa.orm.Session] = {}
        self.inited = False
        self.initing = False
        self.sa_registry = sa.orm.registry()

    def get(self, uid):
        if uid not in self.ms:
            raise Error(f'model {uid!r} not found')
        return self.ms[uid]

    def get_model(self, uid):
        return self.get(uid).model

    def get_class(self, uid):
        return self.get(uid).cls

    def get_keys(self, uid):
        return self.get(uid).keys

    @property
    def models(self):
        ls = []
        for m in self.ms.values():
            ls.append(m.model)
        return ls

    @property
    def fields(self):
        ls = []
        for mod in self.models:
            ls.extend(mod.fields)
        return ls

    def init(self):
        if self.inited:
            return self

        if self.initing:
            raise Error('circular init!')

        self.initing = True

        for mod in self.root.find_all(Object):
            m = t.Data(
                model=mod,
                cls=None,
                table=None,
                keys=[],
            )
            self.ms[mod.uid] = m
            gws.log.debug(f'REGISTRY_INIT FOUND:{m.model.uid}')

        for m in self.ms.values():
            gws.log.debug(f'REGISTRY_INIT KEYS:{m.model.uid}')
            cols = []
            for f in m.model.fields:
                if f.is_primary_key:
                    f.sa_columns(cols)
            m.keys = cols

        for m in self.ms.values():
            gws.log.debug(f'REGISTRY_INIT TABLE:{m.model.uid}')
            cols = []
            for f in m.model.fields:
                if not f.is_primary_key:
                    f.sa_columns(cols)
            cols.extend(m.keys)
            m.table = self.create_table(m.model.get_table().name, cols)

        for m in self.ms.values():
            gws.log.debug(f'REGISTRY_INIT CLASS:{m.model.uid}')
            m.cls = type(f'_SA_{m.model.uid}', (SaBase,), {})
            self.sa_registry.map_imperatively(m.cls, m.table)

        for m in self.ms.values():
            gws.log.debug(f'REGISTRY_INIT PROPS:{m.model.uid}')
            props = {}
            for f in m.model.fields:
                f.sa_properties(props)
            for k, v in props.items():
                getattr(m.cls, '__mapper__').add_property(k, v)

        self.initing = False
        self.inited = True
        return self

    def create_table(self, name, cols):
        metadata = self.sa_registry.metadata
        if name in metadata.tables:
            gws.log.info(f'TABLE {name} already exists')
            return metadata.tables[name]

        schema = 'public'
        if '.' in name:
            schema, name = name.split('.')

        return sa.Table(name, metadata, *cols, schema=schema)

    def session(self, db):
        uid = db.uid

        if uid not in self.engines:
            gws.log.debug(f'ENGINE CREATE {uid}')
            p = db.connect_params
            self.engines[uid] = sa.create_engine('postgresql://', connect_args=p, echo=False, future=True, pool_pre_ping=True)

        if uid not in self.sessions:
            gws.log.debug(f'SESSION CREATE {uid}')
            self.sessions[uid] = sa.orm.Session(self.engines[uid], future=True)

        return self.sessions[uid]

    def commit_all(self):
        for uid, sess in self.sessions.items():
            gws.log.debug(f'SESSION COMMIT {uid}')
            sess.commit()

    def rollback_all(self):
        for uid, sess in self.sessions.items():
            gws.log.debug(f'SESSION ROLLBACK {uid}')
            sess.rollback()

    def close_all(self):
        for uid, sess in self.sessions.items():
            gws.log.debug(f'SESSION CLOSE {uid}')
            sess.close()
        self.sessions = {}


def registry() -> ModelRegistry:
    return gws.get_global(f'model_registry', lambda: ModelRegistry(gws.config.root()))


#:export
class ModelSession:
    def open(self):
        registry().init()

    def commit(self):
        registry().commit_all()

    def rollback(self):
        registry().rollback_all()

    def close(self):
        registry().close_all()


session = ModelSession()


def get(uid: str) -> t.Optional[t.IModel]:
    return registry().get_model(uid)
