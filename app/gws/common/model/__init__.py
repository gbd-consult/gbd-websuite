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
                    'value': q.get('value'),
                    'text': q.get('text') or str(q.get('value')),
                }
                for q in p
            ]

    @property
    def props(self):
        return WidgetProps(
            type=self.type,
            options=self.options,
        )


class StringWidget(Widget):
    pass


class RelationListWidget(Widget):
    pass


class RelationSelectWidget(Widget):
    pass


class FileWidget(Widget):
    pass


class DateWidget(Widget):
    pass


class TextboxWidget(Widget):
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
        pass


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

class FieldValueConfig(t.WithType):
    value: t.Optional[t.Any]
    expression: t.Optional[str]


class FieldRelationConfig(t.Config):
    modelUid: str
    fieldName: str
    title: str
    discriminatorValue: str


class FieldConfig(t.WithType):
    name: str
    title: t.Optional[str]

    widget: t.Optional[WidgetConfig]

    relatedModelUid: t.Optional[str]
    relatedFieldName: t.Optional[str]

    relations: t.Optional[t.List[FieldRelationConfig]]

    foreignKeyName: t.Optional[str]
    foreignKeyType: t.Optional[str]

    discriminatorName: t.Optional[str]
    discriminatorType: t.Optional[str]

    linkTableName: t.Optional[str]
    linkKeyName: t.Optional[str]

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
    relation_type: str
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

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, exclude, depth):
        pass

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps):
        pass

    def read_from_orm(self, fe: t.IFeature, obj, exclude, depth):
        pass

    def write_to_orm(self, fe: t.IFeature, obj):
        pass

    def validate(self, attributes, errors):
        value = attributes.get(self.name)

        if value is None:
            if self.is_required:
                errors.append(t.FeatureError(name=self.name, error='validationErrorNull'))
            return

        if not self.validators:
            err = self.validate_value(value, attributes)
            if err:
                errors.append(t.FeatureError(name=self.name, error=err))
                return

        for v in self.validators:
            err = v.validate_value(self, value, attributes)
            if err:
                errors.append(t.FeatureError(name=self.name, error=err))

    def validate_value(self, value, attributes):
        pass

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

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, exclude, depth):
        val = props.attributes.get(self.name)
        if val is not None:
            fe.attributes[self.name] = val
            return
        val = self.get_default()
        if val is not None:
            fe.attributes[self.name] = val
            return

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val
            return
        val = self.get_default()
        if val is not None:
            props.attributes[self.name] = val
            return

    def read_from_orm(self, fe: t.IFeature, obj, exclude, depth):
        val = getattr(obj, self.name, None)
        if val is not None:
            fe.attributes[self.name] = val
            return
        val = self.get_default()
        if val is not None:
            fe.attributes[self.name] = val
            return

    def write_to_orm(self, fe: t.IFeature, obj):
        val = fe.attributes.get(self.name)
        if val is not None:
            setattr(obj, self.name, val)
            return
        val = self.get_default()
        if val is not None:
            setattr(obj, self.name, val)


class StringField(ScalarField):
    data_type = 'string'

    def sa_adapt_select(self, state):
        ok = (
                state.args.keyword
                and state.keyword_columns
                and self.name in state.keyword_columns)

        if ok:
            state.keyword_conds.append((
                getattr(self.model.get_class(), self.name)
                    .ilike('%' + _escape_like(state.args.keyword) + '%', escape='\\')))

    def validate_value(self, value, attributes):
        v = str(value).strip()
        if not v and self.is_required:
            return 'validationErrorNull'
        if self.min_len is not None and len(v) < self.min_len:
            return 'validationErrorStringTooShort'
        if self.max_len is not None and len(v) > self.max_len:
            return 'validationErrorStringTooLong'
        if self.pattern and not re.match(str(self.pattern), v):
            return 'validationErrorPattern'


class IntegerField(ScalarField):
    data_type = 'integer'

    def validate_value(self, value, attributes):
        try:
            v = int(value)
        except:
            return 'validationErrorInvalidInteger'
        if self.min_value is not None and v < self.min_value:
            return 'validationErrorNumberTooSmall'
        if self.max_value is not None and v > self.max_value:
            return 'validationErrorNumberTooBig'


class FloatField(ScalarField):
    data_type = 'float'

    def validate_value(self, value, attributes):
        try:
            v = float(value)
        except:
            return 'validationErrorInvalidFloat'
        if self.min_value is not None and v < self.min_value:
            return 'validationErrorNumberTooSmall'
        if self.max_value is not None and v > self.max_value:
            return 'validationErrorNumberTooBig'


class DateField(ScalarField):
    data_type = 'date'


def _is_record_with_type(val, type):
    return isinstance(val, (dict, t.Data)) and val.get('type') == type


class FileField(ScalarField):
    data_type = 'string'

    def store_file(self, name, content):
        p = gws.tools.os2.parse_path(name)
        file_name = gws.as_uid(p['name']) + '.' + gws.as_uid(p['extension'])
        file_path = self.var('basePath') + '/' + file_name
        gws.write_file_b(file_path, content)
        return file_name

    def get_file(self, fe: t.IFeature):
        file_name = fe.attr(self.name)
        file_path = self.var('basePath') + '/' + file_name
        return t.FileResponse(
            path=file_path
        )

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, exclude, depth):
        val = props.attributes.get(self.name)
        if _is_record_with_type(val, 'File'):
            fe.attributes[self.name] = val

    def read_from_orm(self, fe: t.IFeature, obj, exclude, depth):
        val = getattr(obj, self.name, None)
        if val is not None:
            fe.attributes[self.name] = val

    def write_to_orm(self, fe: t.IFeature, obj):
        val = fe.attributes.get(self.name)
        if _is_record_with_type(val, 'File'):
            res = self.store_file(val.get('name'), val.get('content'))
            setattr(obj, self.name, res)


class GeometryField(ScalarField):
    data_type = 'string'
    geometry_type = ''

    def sa_adapt_select(self, state):
        if state.args.shape:
            shape = state.args.shape.tolerance_polygon(state.args.map_tolerance)
            shape = shape.transformed_to(self.model.get_table().geometry_crs)
            state.geometry_conds.append(sa.func.st_intersects(
                getattr(self.model.get_class(), self.name),
                sa.cast(shape.ewkb_hex, geosa.Geometry())))

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, exclude, depth):
        val = props.attributes.get(self.name)
        if val is not None:
            fe.attributes[self.name] = gws.gis.shape.from_props(val)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.props

    def read_from_orm(self, fe: t.IFeature, obj, exclude, depth):
        val = getattr(obj, self.name, None)
        if val is not None:
            fe.attributes[self.name] = gws.gis.shape.from_wkb_hex(val)

    def write_to_orm(self, fe: t.IFeature, obj):
        val = fe.attributes.get(self.name)
        if val:
            setattr(obj, self.name, val.ewkb_hex)


##


# def as_feature(s) -> t.IFeature:
#     if s and isinstance(s, t.IFeature):
#         return s
#
#
# def as_feature_collection(s) -> t.IFeatureCollection:
#     if s and isinstance(s, t.IFeatureCollection):
#         return s
#
#
# def as_feature_props(s) -> t.FeatureProps:
#     if s:
#         if isinstance(s, dict):
#             s = t.Data(s)
#         if isinstance(s, t.Data) and s.type == 'Feature':
#             return t.cast(t.FeatureProps, s)
#
#
# def as_feature_collection_props(s) -> t.FeatureCollectionProps:
#     if s:
#         if isinstance(s, dict):
#             s = t.Data(s)
#         if isinstance(s, t.Data) and s.type == 'FeatureCollection':
#             return t.cast(t.FeatureCollectionProps, s)


##


class RelationField(Field):
    def configure(self):
        super().configure()

        self.relations = []

        p = self.var('relatedModelUid')
        if p:
            self.relations.append(t.Data(
                model_uid=self.var('relatedModelUid'),
                field_name=self.var('relatedFieldName'),
            ))

        self.link_table_name = self.var('linkTableName')
        self.link_key_name = self.var('linkKeyName')

        self.foreign_key_name = self.var('foreignKeyName')
        self.foreign_key_type = self.var('foreignKeyType')

        self.discriminator_name = self.var('discriminatorName')
        self.discriminator_type = self.var('discriminatorType')

        p = self.var('relations')
        if p:
            for r in p:
                self.relations.append(t.Data(
                    model_uid=r.modelUid,
                    field_name=r.fieldName,
                    title=r.title,
                    discriminator_value=r.discriminatorValue,
                ))

    @property
    def props(self):
        p = super().props
        p.relations = []
        for r in self.relations:
            p.relations.append(t.Props(
                type=self.relation_type,
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

    ##

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, exclude, depth):
        if depth <= 0:
            return

        val = props.attributes.get(self.name)
        if val is None:
            return

        rel_model = self.first_related_model()

        if self.data_type == 'feature':
            uid = val.get('attributes').get(rel_model.key_name)
            fe.attributes[self.name] = rel_model.get_feature(uid, exclude, depth - 1)

        if self.data_type == 'featureList':
            uids = [f.get('attributes').get(rel_model.key_name) for f in val]
            fe.attributes[self.name] = [rel_model.get_feature(uid, exclude, depth - 1) for uid in uids]

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps):
        if self.data_type == 'feature':
            fe2 = fe.attributes.get(self.name)
            if fe2:
                props.attributes[self.name] = fe2.props

        if self.data_type == 'featureList':
            flist = fe.attributes.get(self.name)
            if flist:
                props.attributes[self.name] = [f.props for f in flist]

    def read_from_orm(self, fe: t.IFeature, obj, exclude, depth):
        if depth <= 0:
            return

        rel_model = self.first_related_model()

        if self.data_type == 'feature':
            obj2 = getattr(obj, self.name, None)
            if obj2:
                exclude = (exclude or []) + [self.relations[0].field_name]
                fe.attributes[self.name] = rel_model.feature_from_orm(obj2, exclude, depth - 1)

        if self.data_type == 'featureList':
            objlist = getattr(obj, self.name, None)
            if objlist:
                exclude = (exclude or []) + [self.relations[0].field_name]
                fe.attributes[self.name] = [
                    rel_model.feature_from_orm(o, exclude, depth - 1)
                    for o in objlist
                ]

    def write_to_orm(self, fe: t.IFeature, obj):

        rel_model = self.first_related_model()

        if self.data_type == 'feature':
            fe2 = fe.attributes.get(self.name)
            if fe2:
                setattr(obj, self.name, rel_model.get_object(fe2.uid))

        if self.data_type == 'featureList':
            flist = fe.attributes.get(self.name)
            if flist:
                setattr(obj, self.name, [rel_model.get_object(f.uid) for f in flist])


##


"""
moRelation   M->1

    HOUSE -> street_id -> STREET

    HOUSE.street:      
        type = STREET
        relatedModel = STREET (parent)
        relatedField (optional) = 1->M field in parent = STREET.houses
        foreignKey = foreign key name (street_id)
"""


class Field_moRelation(RelationField):
    data_type = 'feature'
    relation_type = 'mo'

    sa_adapt_select = RelationField.sa_adapt_select_selectin

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
omRelation   1->M
    
    STREET -> HOUSE (list)

    STREET.houses:
        type = Array<HOUSE>
        relatedModel = HOUSE (child)
        relatedField (mandatory) = HOUSE.street (M->1 field in child)

"""


class Field_omRelation(RelationField):
    data_type = 'featureList'
    relation_type = 'om'

    sa_adapt_select = RelationField.sa_adapt_select_selectin

    def sa_properties(self, properties):
        rel_model = self.first_related_model()
        rel_cls = rel_model.get_class()
        kwargs = {}
        kwargs['back_populates'] = self.relations[0].field_name
        properties[self.name] = sa.orm.relationship(rel_cls, **kwargs)


"""
omMultiRelation   1->M 

    STREET.id = HOUSE.street_id
    STREET.id = TREE.street_id
    
    STREET.objects
        type = Array<HOUSE|TREE>
        relations = [
            { modelUid: HOUSE title 'House' fieldName: street } 
            { modelUid: TREE  title 'Tree' fieldName: street } 
        ]
        
    symmetric = moRelation
"""


class Field_omMultiRelation(RelationField):
    data_type = 'featureList'
    relation_type = 'om'

    def relation_for_model(self, model):
        for r in self.relations:
            if r.model_uid == model.uid:
                return r

    def sa_properties(self, properties):
        for r in self.relations:
            rel_model = registry().get_model(r.model_uid)
            rel_cls = rel_model.get_class()
            kwargs = {}
            kwargs['back_populates'] = r.field_name
            key = self.name + ':' + r.model_uid
            properties[key] = sa.orm.relationship(rel_cls, **kwargs)

    def read_from_orm(self, fe: t.IFeature, obj, exclude, depth):
        if depth <= 0:
            return

        flist = []

        for r in self.relations:
            key = self.name + ':' + r.model_uid
            val = getattr(obj, key, None)
            if val is None:
                continue
            exclude = (exclude or []) + [r.field_name]
            rel_model = registry().get_model(r.model_uid)
            for obj2 in val:
                flist.append(rel_model.feature_from_orm(obj2, exclude, depth - 1))

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


class Field_mmLinkRelation(RelationField):
    data_type = 'featureList'
    relation_type = 'mm'

    sa_adapt_select = RelationField.sa_adapt_select_selectin

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


def _get_uid(val):
    if isinstance(val, (int, str)):
        return val
    if _is_record_with_type(val, 'Feature'):
        return val.get('attributes').get(val.get('keyName'))
    if isinstance(val, t.IFeature):
        return val.uid


def _generic_feature(uid):
    fe = _GENERIC_MODEL.new_feature()
    fe.key_name = 'id'
    fe.attributes = {'id': uid}
    return fe


"""
omGenericRelation   1->M 
    
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


class Field_omGenericRelation(RelationField):
    data_type = 'featureList'
    relation_type = 'om'

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
moGenericRelation   M->1 
    
    https://docs.sqlalchemy.org/en/14/_modules/examples/generic_associations/generic_fk.html
    
    IMAGE.object_id -> STREET.id 
    IMAGE.object_id -> HOUSE.id 

    IMAGE.object
        type = GenericFeature (=only id)
"""


class Field_moGenericRelation(RelationField):
    data_type = 'feature'
    relation_type = 'mo'

    def sa_columns(self, columns):
        col = sa.Column(self.foreign_key_name, _SCALAR_TYPES[self.foreign_key_type])
        columns.append(col)

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, exclude, depth):
        val = props.attributes.get(self.name)
        if val is None:
            return

        uid = _get_uid(val)
        if uid:
            fe.attributes[self.name] = _generic_feature(uid)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.props
            return

    def read_from_orm(self, fe: t.IFeature, obj, exclude, depth):
        val = getattr(obj, self.foreign_key_name, None)
        if val is not None:
            fe.attributes[self.name] = _generic_feature(val)
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


class Field_omDiscriminatedRelation(RelationField):
    data_type = 'featureList'
    relation_type = 'om'

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
            getattr(rel_cls, rel_field.discriminator_name) == rel.discriminator_value
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
                setattr(o, rel_field.discriminator_name, rel.discriminator_value)


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
            { uid: STREET title 'Street' fieldName: objects discriminatorValue: 'street' } 
            { uid: HOUSE  title 'House'  fieldName: objects discriminatorValue: 'house'  } 
        ]
        
"""


class Field_moDiscriminatedRelation(RelationField):
    data_type = 'feature'
    relation_type = 'mo'

    def configure(self):
        super().configure()
        self.relations = []
        for p in self.var('relations'):
            self.relations.append(t.Data(
                model_uid=p.uid,
                field_name=p.fieldName,
                discriminator_value=p.discriminatorValue
            ))

    def relation_for_model(self, model):
        for r in self.relations:
            if r.model_uid == model.uid:
                return r

    def relation_for_discriminator(self, dv):
        for r in self.relations:
            if r.discriminator_value == dv:
                gws.log.debug(f'found model {r.model_uid!r} for dv={dv!r}')
                return r

    def sa_columns(self, columns):
        col = sa.Column(self.foreign_key_name, _SCALAR_TYPES[self.foreign_key_type])
        columns.append(col)
        col = sa.Column(self.discriminator_name, _SCALAR_TYPES[self.discriminator_type])
        columns.append(col)

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, exclude, depth):
        val = props.attributes.get(self.name)
        if val is not None:
            rel_model = registry().get_model(val.modelUid)
            uid = val.attributes.get(rel_model.key_name)
            fe.attributes[self.name] = rel_model.feature_from_uid2(uid)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.props

    def read_from_orm(self, fe: t.IFeature, obj, exclude, depth):
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
            setattr(obj, self.discriminator_name, rel.discriminator_value)
            gws.log.debug(f'@@ {self.foreign_key_name}={val.uid!r}, {self.discriminator_name}={rel.discriminator_value!r}')


##

_WIDGET_TYPES = {
    'combo': ComboWidget,
    'date': DateWidget,
    'file': FileWidget,
    'float': FloatWidget,
    'geometry': GeometryWidget,
    'integer': IntegerWidget,
    'readonly': ReadonlyWidget,
    'relationList': RelationListWidget,
    'relationSelect': RelationSelectWidget,
    'select': SelectWidget,
    'string': StringWidget,
    'textbox': TextboxWidget,
    'measurement': MeasurementWidget,
}

_FIELD_TYPES = {
    'string': StringField,
    'integer': IntegerField,
    'float': FloatField,
    'date': DateField,
    'file': FileField,

    'geometry': GeometryField,

    'omRelation': Field_omRelation,
    'omMultiRelation': Field_omMultiRelation,
    'moRelation': Field_moRelation,
    'mmLinkRelation': Field_mmLinkRelation,
    'omGenericRelation': Field_omGenericRelation,
    'moGenericRelation': Field_moGenericRelation,
    'omDiscriminatedRelation': Field_omDiscriminatedRelation,
    'moDiscriminatedRelation': Field_moDiscriminatedRelation,
}

_VALIDATOR_TYPES = {
}


##


class ModelRegistry:
    def __init__(self, root):
        self.root = root
        self.ms = {}
        self.engines = {}
        self.sessions = {}
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
            m.cls = type(f'_SA_{m.model.uid}', (), {})
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

    def session(self, uid):
        db = self.ms[uid].model.get_db()
        if not db:
            return

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
    def commit(self):
        registry().commit_all()

    def rollback(self):
        registry().rollback_all()

    def close(self):
        registry().close_all()

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.close()


def session():
    return ModelSession()


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
    fields: t.List[t.IModelField]
    key_name: str
    geometry_name: str

    def configure(self):
        super().configure()

        self.layer = None

        self.key_name = ''
        self.geometry_name = ''

        self.sql_filter = self.var('sqlFilter')

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
        registry().init()

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

    def save(self, fe: t.IFeature, is_new: bool):
        registry().init()

        if is_new:
            obj = self.get_class()()
        else:
            obj = self.get_object(fe.uid)

        for f in self.fields:
            f.write_to_orm(fe, obj)

        if is_new:
            self.sa_session().add(obj)

        fe._model_data = {'object': obj}
        return fe

    def delete(self, fe: t.IFeature):
        registry().init()
        self.sa_session().delete(self.get_object(fe.uid))

    def reload(self, fe: t.IFeature, depth=0):
        md = getattr(fe, '_model_data', None)
        if md:
            f2 = self.feature_from_orm(md['object'], depth=depth)
            fe.attributes = f2.attributes
        return fe

    def validate(self, fe) -> t.List[t.FeatureError]:
        errors = []
        for f in self.fields:
            f.validate(fe.attributes, errors)
        return errors

    def new_feature(self):
        registry().init()
        fe = gws.gis.feature.Feature()
        fe.model = self
        fe.key_name = self.key_name
        fe.geometry_name = self.geometry_name
        if self.layer:
            fe.layer = self.layer
        return fe

    def get_feature(self, uid, exclude=None, depth=0):
        obj = self.get_object(uid)
        if obj:
            return self.feature_from_orm(obj, exclude, depth)

    def feature_from_uid2(self, props: t.FeatureProps, exclude=None, depth=0):
        fe = self.new_feature()
        uid = props.attributes.get(self.key_name)
        obj = self.get_object(uid)
        return self.feature_from_orm(obj, exclude, depth)

    def feature_from_props(self, props: t.FeatureProps, exclude=None, depth=0):
        fe = self.new_feature()
        for f in self.fields:
            f.read_from_props(fe, props, exclude, depth)
        return fe

    def feature_from_orm(self, obj, exclude=None, depth=0):
        fe = self.new_feature()

        for f in self.fields:
            if exclude and f.name in exclude:
                continue
            f.read_from_orm(fe, obj, exclude, depth)

        setattr(fe, '_model_data', {'object': obj})
        return fe

    def feature_props(self, fe):
        registry().init()
        props = t.FeatureProps(
            type='Feature',
            attributes={},
            keyName=self.key_name,
            geometryName=self.geometry_name,
            modelUid=self.uid,
            layerUid=self.layer.uid if self.layer else None,
        )
        for f in self.fields:
            f.write_to_props(fe, props)
        return props

    ##

    def get_db(self):
        if self.layer:
            return getattr(self.layer, 'db', None)

    def get_table(self) -> t.SqlTable:
        if self.layer:
            return getattr(self.layer, 'table', None)

    def get_field(self, name):
        for f in self.fields:
            if f.name == name:
                return f

    def get_class(self):
        return registry().get_class(self.uid)

    def get_keys(self):
        return registry().get_keys(self.uid)

    def get_object(self, uid):
        return self.sa_session().get(self.get_class(), uid)

    ##

    def sa_session(self):
        return registry().session(self.uid)

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
            state.sel = state.sel.where(sa.text(args.extra_where[0], *args.extra_where[1:]))

        if self.sql_filter:
            state.sel = state.sel.where(sa.text(self.sql_filter))

        gws.log.debug(f'SA_MAKE_SELECT: {str(state.sel)}')

        return state.sel

    @property
    def props(self):
        return Props(
            uid=self.uid,
            layerUid=self.layer.uid,
            keyName=self.key_name,
            geometryName=self.geometry_name,
            fields=[f.props for f in self.fields]
        )


class GenericModel(Object):
    key_name = 'id'
    geometry_name = ''
    layer = None

    def feature_from_props(self, props: t.FeatureProps, exclude=None, depth=0):
        fe = self.new_feature()
        fe.attributes['id'] = gws.get(props, 'attributes.id')
        return fe

    def feature_props(self, fe):
        props = t.FeatureProps(
            type='Feature',
            attributes={},
            keyName='id',
            geometryName='',
            layerUid=None,
        )
        return props


_GENERIC_MODEL = GenericModel()


def _escape_like(s, escape='\\'):
    return (
        s
            .replace(escape, escape + escape)
            .replace('%', escape + '%')
            .replace('_', escape + '_'))
