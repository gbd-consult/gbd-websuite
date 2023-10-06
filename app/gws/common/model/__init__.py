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

#:export ModelContext
class ModelContext(t.Data):
    mode: str
    user: t.IUser
    depth: int
    errors: t.List[t.FeatureError]
    access: str
    project: t.IProject




class ModelPermissionsConfig(t.Config):
    read: t.Optional[t.WithAccess]
    write: t.Optional[t.WithAccess]
    create: t.Optional[t.WithAccess]
    delete: t.Optional[t.WithAccess]


class FieldPermissionsConfig(t.Config):
    read: t.Optional[t.WithAccess]
    write: t.Optional[t.WithAccess]


class Permssion(t.Data):
    access: t.List[t.Access]


class ModelPermissions(t.Data):
    read: Permssion
    write: Permssion
    create: Permssion
    delete: Permssion
    parent: Permssion


class FieldPermissions(t.Data):
    read: Permssion
    write: Permssion
    parent: Permssion


##


class WidgetConfig(t.WithType):
    items: t.Optional[t.List[t.Any]]
    fileField: t.Optional['FieldNameTypeConfig']
    search: t.Optional[str]
    previewUrl: t.Optional[str]
    downloadUrl: t.Optional[str]


class WidgetProps(t.Props):
    type: str
    options: dict
    readOnly: bool = False


#:export IModelWidget
class Widget(gws.Object, t.IModelWidget):
    type: str

    def configure(self):
        super().configure()
        self.type = self.var('type')

        self.options = {}
        for k, v in vars(self.config).items():
            if k not in {'type', 'uid'}:
                self.options[k] = v

        self.options['items'] = self.var('items')

        p = self.var('fileField')
        if p:
            self.options['fileFieldName'] = gws.get(p, 'name')

    @property
    def props(self):
        return WidgetProps(
            type=self.type,
            options=self.options,
            readOnly=False,
        )


##

VALIDATION_ERROR_PREFIX = 'validationError'


class ValidatorConfig(t.WithType):
    message: str
    operation: t.Optional[str]
    value: t.Optional['ValueConfig']


#:export IModelValidator
class Validator(gws.Object, t.IModelValidator):
    def configure(self):
        super().configure()

        self.type: str = self.var('type')
        self.message: str = self.var('message')

    def validate(self, field: t.IModelField, value: t.Any, fe: t.IFeature):
        pass


class Validator_string(Validator):
    def validate(self, field, value, fe):
        v = str(value).strip()
        if len(v) == 0 and field.is_required:
            raise ValidationError('Null')
        fe.attributes[field.name] = v


class Validator_integer(Validator):
    def validate(self, field, value, fe):
        try:
            fe.attributes[field.name] = int(value)
        except ValueError:
            raise ValidationError('InvalidInteger')


class Validator_float(Validator):
    def validate(self, field, value, fe):
        try:
            fe.attributes[field.name] = float(value)
        except ValueError:
            raise ValidationError('InvalidFloat')


class Validator_geometryConstraint(Validator):
    def validate(self, field, value, fe):
        this = t.cast(gws.gis.shape.Shape, value)

        v = eval_value(fe, self.var('value'), None)
        if isinstance(v, dict):
            other = gws.gis.shape.from_geometry(v, this.crs)
        elif isinstance(v, str):
            other = gws.gis.shape.from_wkt(v, this.crs)
        else:
            other = t.cast(gws.gis.shape.Shape, v)

        other = other.transformed_to(this.crs)

        op = self.var('operation')
        if op == 'within' and not this.within(other):
            raise ValidationError('GeometryConstraint')
        if op == 'intersects' and not this.intersects(other):
            raise ValidationError('GeometryConstraint')


##


def eval_value(fe: t.IFeature, val: 'ValueConfig', env):
    if val.type == 'static':
        return val.value
    if val.type == 'expression':
        gws.log.debug(f'evaluating {val.expression=}')
        return eval(val.expression)


##

class ValidationError(Exception):
    pass


class ValueConfig(t.WithType):
    value: t.Optional[t.Any]
    expression: t.Optional[str]


class FieldValueConfig(t.Config):
    read: t.Optional[ValueConfig]
    write: t.Optional[ValueConfig]
    readDefault: t.Optional[ValueConfig]
    writeDefault: t.Optional[ValueConfig]
    serverDefault: t.Optional[str]


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


class FieldTextSearchConfig:
    type: str
    caseSensitive: bool = False
    minLength: int = 0


class FieldConfig(t.WithType):
    name: str
    title: t.Optional[str]

    widget: t.Optional[WidgetConfig]

    relation: t.Optional[FieldRelationConfig]
    relations: t.Optional[t.List[FieldRelationConfig]]

    foreignKey: t.Optional[FieldNameTypeConfig]
    discriminatorKey: t.Optional[FieldNameTypeConfig]
    link: t.Optional[FieldLinkConfig]

    filePath: t.Optional[str]

    value: t.Optional[FieldValueConfig]
    validators: t.Optional[t.List[ValidatorConfig]]

    geometryType: t.Optional[str]

    errorMessages: t.Optional[dict]

    isRequired: bool = False
    isUnique: bool = False
    isPrimaryKey: bool = False

    permissions: t.Optional[FieldPermissionsConfig]
    textSearch: t.Optional[FieldTextSearchConfig]


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
    title: str
    widget: t.Optional[t.IModelWidget]
    validators: t.List[t.IModelValidator]
    data_type: str
    geometry_type: str
    permissions: FieldPermissions
    value: t.Data
    error_messages: t.Dict[str, str]
    text_search: t.Data

    def configure(self):
        super().configure()

        self.type = self.var('type')
        self.name = self.var('name')
        self.title = self.var('title', default=self.name)
        self.geometry_type = self.var('geometryType', default='').upper()

        self.value = self.var('value') or t.Data()

        self.is_primary_key: bool = self.var('isPrimaryKey')
        self.is_required: bool = self.var('isRequired')
        self.is_searchable: bool = self.var('isSearchable')
        self.is_unique: bool = self.var('isUnique')

        self.error_messages = self.var('errorMessages', default={})

        self.widget = None
        p = self.var('widget')
        if p:
            cls = get_class('widget', p.type)
            uid = gws.sha256(repr(p))
            wgt = t.cast(t.IModelWidget, self.root.create_shared_object(cls, uid, p))
            self.widget = wgt

        self.validators = []
        ps = self.var('validators')
        if ps:
            for p in ps:
                cls = get_class('validator', p.type)
                v = t.cast(t.IModelValidator, self.create_child(cls, p))
                v.field = self
                self.validators.append(v)

        self.permissions = FieldPermissions(
            read=self.var('permissions.read') or t.Data(),
            write=self.var('permissions.write') or t.Data(),
        )

        self.text_search = self.var('textSearch')

    def props_for(self, user):
        p = super().props_for(user)
        if gws.get(p, 'widget') and not user.can_use(self.permissions.write):
            p['widget']['readOnly'] = True
        return p

    def sa_columns(self, columns):
        pass

    def sa_properties(self, properties):
        pass

    def sa_adapt_select(self, state):
        pass

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        pass

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        pass

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        pass

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        pass

    def apply_value(self, fe: t.IFeature, kind, mc: t.ModelContext):
        val = gws.get(self.value, mc.access + ('Default' if kind == 'default' else ''))
        if val is not None and (kind == 'fixed' or fe.attr(self.name) is None):
            fe.attributes[self.name] = eval_value(fe, val, mc)
            return True
        return False

    def prepend_validator(self, cfg):
        for v in self.validators:
            if v.type == cfg.type:
                return
        cls = get_class('validator', cfg.type)
        v = t.cast(t.IModelValidator, self.create_child(cls, cfg))
        self.validators.insert(0, v)

    def validate(self, fe: t.IFeature, mc: t.ModelContext):
        try:
            if fe.attr(self.name) is None:
                if self.is_required:
                    raise ValidationError('Null')
                return
            for v in self.validators:
                v.validate(self, fe.attr(self.name), fe)
        except ValidationError as err:
            msg = err.args[0]
            if msg in self.error_messages:
                msg = self.error_messages[msg]
            else:
                msg = VALIDATION_ERROR_PREFIX + msg
            mc.errors.append(t.FeatureError(fieldName=self.name, message=msg))

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
        kwargs = {}
        if self.is_primary_key:
            kwargs['primary_key'] = True
        if self.value.serverDefault:
            kwargs['server_default'] = sa.text(self.value.serverDefault)

        col = sa.Column(self.name, _SCALAR_TYPES[self.data_type], **kwargs)
        columns.append(col)

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        val = props.attributes.get(self.name)
        if val is not None:
            fe.attributes[self.name] = val

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        val = getattr(obj, self.name, None)
        if val is not None:
            fe.attributes[self.name] = val

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        if self.name in fe.attributes:
            setattr(obj, self.name, fe.attributes[self.name])

    def sa_adapt_select(self, state):
        if state.args.keyword and self.text_search:

            ts = self.text_search

            if ts.minLength and len(state.args.keyword) < ts.minLength:
                return

            model = t.cast(t.IDbModel, self.model)
            fld = sa.sql.cast(
                getattr(model.get_class(), self.name),
                sa.String)

            kw = _escape_like(state.args.keyword)

            if ts.type == 'like':
                kw = '%' + kw + '%'
            if ts.type == 'begin':
                kw = kw + '%'
            if ts.type == 'end':
                kw = '%' + kw

            if ts.caseSensitive:
                state.keyword_conds.append(fld.like(kw, escape='\\'))
            else:
                state.keyword_conds.append(fld.ilike(kw, escape='\\'))


class Field_string(ScalarField):
    data_type = 'string'

    def configure(self):
        super().configure()
        self.prepend_validator(t.Data(type='string'))


class Field_integer(ScalarField):
    data_type = 'integer'

    def configure(self):
        super().configure()
        self.prepend_validator(t.Data(type='integer'))

    # def validate(self, fe: t.IFeature, errors):
    #     try:
    #         v = self.convert(value)
    #     except:
    #         raise ValidationError('validationErrorInvalidInteger')
    #     if v is None:
    #         if self.is_required:
    #             raise ValidationError('validationErrorNull')
    #         return v
    #     if self.min_value is not None and v < self.min_value:
    #         raise ValidationError('validationErrorNumberTooSmall')
    #     if self.max_value is not None and v > self.max_value:
    #         raise ValidationError('validationErrorNumberTooBig')
    #     return v


class Field_float(ScalarField):
    data_type = 'float'
    min_value = None
    max_value = None

    def configure(self):
        super().configure()
        self.prepend_validator(t.Data(type='float'))

    def convert(self, val):
        if isinstance(val, float):
            return val
        try:
            return float(val)
        except:
            raise ValueError('invalid float')

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


class Field_date(ScalarField):
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


class FileValue(t.Data):
    name: str
    path: t.Optional[str]
    content: t.Optional[bytes]
    mime: t.Optional[str]


class Field_file(ScalarField):
    data_type = 'string'

    def normalize_file_name(self, s):
        p = gws.tools.os2.parse_path(s)
        return

    def store_file(self, path, content):
        gws.write_file_b(path, content)

    def name_to_path(self, name: str, fe: t.IFeature):
        p = gws.tools.os2.parse_path(name)
        env = t.Data(
            feature=fe,
            file=t.Data(
                name=gws.as_uid(p['name']),
                extension=gws.as_uid(p['extension'].lower())
            )
        )
        fp = self.var('filePath')
        gws.log.debug(f'name_to_path {fp=}')
        path = eval(fp)
        gws.log.debug(f'name_to_path {path=}')
        return path

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, mc):
        val = props.attributes.get(self.name)
        if val is not None:
            fe.attributes[self.name] = FileValue(
                name=gws.get(val, 'name'),
                content=gws.get(val, 'content'))

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, mc):
        val = fe.attr(self.name)
        if val:
            props.attributes[self.name] = FileValue(
                name=val.name,
                mime=gws.tools.mime.for_path(val.name)
            )

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        path = getattr(obj, self.name, None)
        if path is not None:
            p = gws.tools.os2.parse_path(path)
            fe.attributes[self.name] = FileValue(
                path=path,
                name=p['filename'],
                mime=gws.tools.mime.for_path(path))

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        val = fe.attr(self.name)
        if val is not None:
            name = gws.get(val, 'name')
            content = gws.get(val, 'content')
            if name and content:
                path = self.name_to_path(name, fe)
                self.store_file(path, content)
                setattr(obj, self.name, path)


class Field_geometry(ScalarField):
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

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        val = props.attributes.get(self.name)
        if val is not None:
            fe.attributes[self.name] = gws.gis.shape.from_props(val)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.props

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        val = getattr(obj, self.name, None)
        if val is not None:
            fe.attributes[self.name] = gws.gis.shape.from_wkb_hex(val)

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        val = fe.attributes.get(self.name)
        if val is not None:
            setattr(obj, self.name, val.ewkb_hex)


##


class RelatedField(Field):
    model: t.IDbModel
    link_table_name: str
    link_key_name: str
    foreign_key_name: str
    foreign_key_type: str
    discriminator_name: str
    discriminator_type: str
    relations: t.List[t.Data]

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

    def apply_value(self, fe: t.IFeature, kind, mc: t.ModelContext):
        val = gws.get(self.value, mc.access + ('Default' if kind == 'default' else ''))
        if val is not None and (kind == 'fixed' or fe.attr(self.name) is None):
            v = eval_value(fe, val, mc)
            rel_model = self.first_related_model()
            if rel_model:
                fe.attributes[self.name] = rel_model.get_feature(v, mc)
            return True
        return False

    ##


##


class RelatedFeatureField(RelatedField):
    data_type = 'feature'

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        if mc.depth <= 0:
            return

        val = props.attributes.get(self.name)
        if val is None:
            return

        rel_model = self.first_related_model()
        uid = val.get('attributes', {}).get(rel_model.key_name)
        fe.attributes[self.name] = rel_model.get_feature(uid, clone_mc(mc, depth=mc.depth - 1))

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        fe2 = fe.attributes.get(self.name)
        if fe2:
            props.attributes[self.name] = fe2.model.feature_props(fe2, mc.depth - 1)

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        if mc.depth <= 0:
            return
        rel_model = self.first_related_model()
        obj2 = getattr(obj, self.name, None)
        if obj2:
            # exclude = (exclude or []) + [self.relations[0].field_name]
            fe.attributes[self.name] = rel_model.feature_from_orm(obj2, clone_mc(mc, depth=mc.depth - 1))

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        rel_model = self.first_related_model()
        rel_feature = fe.attributes.get(self.name)
        if rel_feature:
            setattr(obj, self.name, rel_model.get_object(rel_feature.uid))


class RelatedFeatureListField(RelatedField):
    data_type = 'featureList'

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        if mc.depth <= 0:
            return
        val = props.attributes.get(self.name)
        if val is None:
            return

        rel_model = self.first_related_model()
        uids = [f.get('attributes').get(rel_model.key_name) for f in val]
        mc2 = clone_mc(mc, depth=mc.depth - 1)
        fe.attributes[self.name] = [rel_model.get_feature(uid, mc2) for uid in uids]

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        flist = fe.attributes.get(self.name)
        if flist:
            props.attributes[self.name] = [fe2.model.feature_props(fe2, mc.depth - 1) for fe2 in flist]

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        if mc.depth <= 0:
            return

        rel_model = self.first_related_model()

        objlist = getattr(obj, self.name, None)
        if objlist:
            mc2 = clone_mc(mc, depth=mc.depth - 1)
            fe.attributes[self.name] = [
                rel_model.feature_from_orm(o, mc2)
                for o in objlist
            ]

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
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

        own_pk = self.model.get_keys()
        rel_pk = rel_model.get_keys()[0].name
        own_cls = self.model.get_class()

        kwargs['primaryjoin'] = getattr(own_cls, self.foreign_key_name) == getattr(rel_cls, rel_pk)

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

        own_pk = self.model.get_keys()
        rel_fk_name = rel_model.get_field(self.relations[0].field_name).foreign_key_name
        own_cls = self.model.get_class()

        kwargs['primaryjoin'] = getattr(own_cls, own_pk[0].name) == getattr(rel_cls, rel_fk_name)
        kwargs['foreign_keys'] = getattr(rel_cls, rel_fk_name)

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

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        if mc.depth <= 0:
            return

        flist = []

        for r in self.relations:
            key = self.name + ':' + r.model_uid
            val = getattr(obj, key, None)
            if val is None:
                continue
            rel_model = registry().get_model(r.model_uid)

            for obj2 in val:
                flist.append(rel_model.feature_from_orm(obj2, clone_mc(mc, depth=mc.depth-1)))

        if flist:
            fe.attributes[self.name] = flist

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
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

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        val = props.attributes.get(self.name)
        if val is None:
            return

        rel_model = None
        if isinstance(val, (int, str)):
            uid = val
        elif isinstance(val, t.IFeature):
            uid = val.uid
            rel_model = val.model
        else:
            key = gws.get(val, 'keyName')
            uid = gws.get(val, ['attributes', key])
            model_uid = gws.get(val, 'modelUid')
            if model_uid:
                rel_model = registry().get_model(model_uid)

        if uid:
            if rel_model:
                fe.attributes[self.name] = rel_model.get_feature(uid, clone_mc(mc, depth=mc.depth-1))
            else:
                fe.attributes[self.name] = generic_feature(uid=uid)

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.model.feature_props(val, clone_mc(mc, depth=mc.depth-1))
            return

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        val = getattr(obj, self.foreign_key_name, None)
        if val is not None:
            fe.attributes[self.name] = generic_feature(uid=val)
            return

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
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

        rel_field = t.cast(RelatedField, rel_model.get_field(self.relations[0].field_name))

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

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        super().write_to_orm(fe, obj, mc)

        rel_model = self.first_related_model()
        rel_obj_list = getattr(obj, self.name)
        if rel_obj_list:
            rel_field = t.cast(RelatedField, rel_model.get_field(self.relations[0].field_name))
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

    def read_from_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        if mc.depth <= 0:
            return
        val = props.attributes.get(self.name)
        if val is not None:
            rel_model = registry().get_model(val.get('modelUid'))
            uid = val.get('attributes', {}).get(rel_model.key_name)
            fe.attributes[self.name] = rel_model.get_feature(uid, clone_mc(mc, depth=mc.depth-1))

    def write_to_props(self, fe: t.IFeature, props: t.FeatureProps, mc: t.ModelContext):
        val = fe.attributes.get(self.name)
        if val is not None:
            props.attributes[self.name] = val.props

    def read_from_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        val = getattr(obj, self.foreign_key_name, None)
        rel = self.relation_for_discriminator(getattr(obj, self.discriminator_name, None))
        if val is not None:
            rel_model = registry().get_model(rel.model_uid)
            sobj = rel_model.get_object(val)
            fe.attributes[self.name] = rel_model.feature_from_orm(sobj, clone_mc(mc, depth=mc.depth-1))
            return

    def write_to_orm(self, fe: t.IFeature, obj, mc: t.ModelContext):
        val = fe.attributes.get(self.name)
        if val is not None:
            rel = self.relation_for_model(val.model)
            setattr(obj, self.foreign_key_name, val.uid)
            setattr(obj, self.discriminator_name, rel.discriminator)
            gws.log.debug(f'@@ {self.foreign_key_name}={val.uid!r}, {self.discriminator_name}={rel.discriminator!r}')


##


class SortConfig:
    fieldName: str
    reverse: bool = False


class Config(t.WithAccess):
    """Model configuration"""

    type: t.Optional[str]
    fields: t.List[FieldConfig]
    permissions: t.Optional[ModelPermissionsConfig]
    filter: t.Optional[str]
    sort: t.Optional[t.List[SortConfig]]


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
    permissions: ModelPermissions

    def configure(self):
        super().configure()

        self.layer = None

        self.key_name = ''
        self.geometry_name = ''

        self.fields = []

        self.permissions = ModelPermissions(
            read=self.var('permissions.read') or t.Data(),
            write=self.var('permissions.write') or t.Data(),
            create=self.var('permissions.create') or t.Data(),
            delete=self.var('permissions.delete') or t.Data(),
        )

        for p in self.var('fields'):
            cls = get_class('field', p.type)
            f = t.cast(t.IModelField, self.create_child(cls, p))
            f.model = self
            f.permissions.read.parent = self.permissions.read
            f.permissions.write.parent = self.permissions.write
            self.fields.append(f)

            if isinstance(f, Field_geometry) and not self.geometry_name:
                self.geometry_name = f.name
            if f.is_primary_key:
                self.key_name = f.name

    def activate(self):
        try:
            registry().init()
        except Exception as exc:
            gws.log.error('MODEL REGISTRY INIT FAILED')
            gws.log.error(repr(exc))
            raise

    def select(self, args: t.SelectArgs, mc: t.ModelContext) -> t.List[t.IFeature]:
        return []

    def save(self, fe: t.IFeature, mc: t.ModelContext) -> t.IFeature:
        return fe

    def delete(self, fe: t.IFeature, mc: t.ModelContext):
        pass

    def reload(self, fe: t.IFeature, mc: t.ModelContext):
        pass

    def validate(self, fe: t.IFeature, mc: t.ModelContext) -> t.List[t.FeatureError]:
        for f in self.fields:
            f.validate(fe, mc)
        return mc.errors

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

    def get_feature(self, uid, mc: t.ModelContext) -> t.Optional[t.IFeature]:
        return None

    def feature_from_props(self, props: t.FeatureProps, mc: t.ModelContext):
        fe = self.init_feature()
        fe.elements = props.get('elements') or {}
        fe.category = props.get('category') or ''
        fe.is_new = bool(props.get('isNew'))
        for f in self.fields:
            f.read_from_props(fe, props, mc)
        return fe

    def feature_props(self, fe: t.IFeature, mc: t.ModelContext):
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
            f.write_to_props(fe, props, mc)

        if fe.errors:
            props.errors = fe.errors

        return props

    def get_field(self, name: str) -> t.Optional[t.IModelField]:
        for f in self.fields:
            if f.name == name:
                return f

    def props_for(self, user):
        p = super().props_for(user)
        p['fields'] = [f.props_for(user) for f in self.fields]
        return p

    def apply_permissions_and_defaults(self, fe: t.IFeature, mc: t.ModelContext):
        for f in self.fields:
            if f.apply_value(fe, 'fixed', mc):
                continue
            if not mc.user.can_use(f.permissions.get(mc.access)):
                gws.log.debug(f'remove field={f.name!r} mode={mc.access!r}')
                fe.attributes.pop(f.name, None)
            f.apply_value(fe, 'default', mc)
        return fe

    @property
    def props(self):
        p = Props(uid=self.uid)
        if self.layer:
            p.layerUid = self.layer.uid
        if self.key_name:
            p.keyName = self.key_name
        if self.geometry_name:
            p.geometryName = self.geometry_name
        return p


#:export IDbModel
class DbModel(Object, t.IDbModel):
    filter = None
    sort = None

    def configure(self):
        super().configure()
        self.filter = self.var('filter')
        self.sort = self.var('sort', default=[])

    def select(self, args: t.SelectArgs, mc: t.ModelContext) -> t.List[t.IFeature]:
        sel = self.sa_make_select(args)
        if sel is None:
            gws.log.debug('empty select')
            return []

        cls = self.get_class()
        flist = []

        cursor = self.sa_session().execute(sel)
        for row in cursor.unique().all():
            obj = getattr(row, cls.__name__)
            fe = self.feature_from_orm(obj, mc)
            flist.append(fe)

        return flist

    def save(self, fe: t.IFeature, mc: t.ModelContext) -> t.IFeature:
        mc = ModelContext(mode='save', user=mc.user, depth=1,)
        obj = self.orm_object_for_save(fe, mc)
        self.sa_session().commit()
        return self.feature_from_orm(obj, mc)

    def delete(self, fe: t.IFeature, mc: t.ModelContext):
        self.sa_session().delete(self.get_object(fe.uid))

    def reload(self, fe: t.IFeature, mc: t.ModelContext):
        md = getattr(fe, '_model_data', None)
        if md:
            f2 = self.feature_from_orm(md['object'], mc)
            fe.attributes = f2.attributes
        fe.is_new = False
        return fe

    ##

    def orm_object_for_save(self, fe: t.IFeature, mc: t.ModelContext):
        if fe.is_new:
            obj = self.get_class()()
        else:
            obj = self.get_object(fe.uid)

        for f in self.fields:
            f.write_to_orm(fe, obj, mc)

        if fe.is_new:
            self.sa_session().add(obj)

        fe._model_data = {'object': obj}
        return obj

    def get_feature(self, uid, mc: t.ModelContext) -> t.Optional[t.IFeature]:
        obj = self.get_object(uid)
        if obj:
            return self.feature_from_orm(obj, mc)

    def feature_from_orm(self, obj, mc: t.ModelContext):
        fe = self.init_feature()

        # gws.log.debug(f"FETCH {self.uid} uid={getattr(obj, self.key_name, '?')}")

        for f in self.fields:
            f.read_from_orm(fe, obj, mc)

        return fe

    def get_db(self):
        if self.layer:
            return getattr(self.layer, 'db', None)

    def get_table(self) -> t.SqlTable:
        if self.layer:
            return getattr(self.layer, 'table', None)

    def get_sa_table(self):
        return registry().get_table_for_model(self.uid)

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

        for s in self.sort:
            fn = sa.desc if s.reverse else sa.asc
            state.sel = state.sel.order_by(fn(getattr(cls, s.fieldName)))

        if args.limit:
            state.sel = state.sel.limit(args.limit)

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

    def feature_from_props(self, props: t.FeatureProps, mc: t.ModelContext):
        fe = super().feature_from_props(props, mc)
        fe.attributes = gws.get(props, 'attributes', default={})
        if props.geometryName:
            fe.geometry_name = props.geometryName
        gn = fe.geometry_name
        if gn and fe.attributes.get(gn):
            fe.attributes[gn] = gws.gis.shape.from_props(fe.attributes.get(gn))
        return fe

    def feature_props(self, fe: t.IFeature, mc: t.ModelContext):
        props = super().feature_props(fe, mc)
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
    fe = m.init_feature()

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
        self.tables = {}
        self.engines: t.Dict[str, sa.engine.Engine] = {}
        self.sessions: t.Dict[str, sa.orm.Session] = {}
        self.inited = False
        self.initing = False
        self.sa_registry = sa.orm.registry()

    def get(self, uid):
        if uid not in self.ms:
            raise Error(f'model {uid!r} not found')
        return self.ms[uid]

    def get_model(self, uid) -> t.IDbModel:
        return self.get(uid).model

    def get_class(self, uid):
        return self.get(uid).cls

    def get_keys(self, uid):
        return self.get(uid).keys

    def get_table_for_model(self, uid):
        if uid not in self.ms:
            raise Error(f'model {uid!r} not found')
        return self.ms[uid].table

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

        gws.log.set_level('DEBUG')

        gws.log.debug('REGISTRY_INIT')
        self.initing = True

        for mod in self.root.find_all(DbModel):
            m = t.Data(
                model=mod,
                cls=None,
                table=None,
                keys=[],
            )
            self.ms[mod.uid] = m
            # gws.log.debug(f'REGISTRY_INIT FOUND:{m.model.uid}')

        for m in self.ms.values():
            # gws.log.debug(f'REGISTRY_INIT KEYS:{m.model.uid}')
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
            # gws.log.debug(f'REGISTRY_INIT CLASS:{m.model.uid}')
            m.cls = type(f'_SA_{m.model.uid}', (SaBase,), {})
            self.sa_registry.map_imperatively(m.cls, m.table)

        for m in self.ms.values():
            # gws.log.debug(f'REGISTRY_INIT PROPS:{m.model.uid}')
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
        schema = 'public'
        if '.' in name:
            schema, name = name.split('.')

        tab = sa.Table(name, metadata, *cols, schema=schema, extend_existing=True)
        self.tables[name] = tab
        return tab

    def get_table(self, name):
        return self.tables.get(name)

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


##

_TYPES = {
    'widget:checkBox': Widget,
    'widget:comboBox': Widget,
    'widget:dateInput': Widget,
    'widget:documentList': Widget,
    'widget:featureList': Widget,
    'widget:featureSelect': Widget,
    'widget:featureSuggest': Widget,
    'widget:file': Widget,
    'widget:floatInput': Widget,
    'widget:geometry': Widget,
    'widget:input': Widget,
    'widget:integerInput': Widget,
    'widget:measurement': Widget,
    'widget:select': Widget,
    'widget:staticText': Widget,
    'widget:textArea': Widget,

    'field:string': Field_string,
    'field:integer': Field_integer,
    'field:float': Field_float,
    'field:date': Field_date,
    'field:file': Field_file,
    'field:geometry': Field_geometry,
    'field:relatedFeature': Field_relatedFeature,
    'field:relatedFeatureList': Field_relatedFeatureList,
    'field:relatedMultiFeatureList': Field_relatedMultiFeatureList,
    'field:relatedLinkedFeatureList': Field_relatedLinkedFeatureList,
    'field:relatedGenericFeatureList': Field_relatedGenericFeatureList,
    'field:relatedGenericFeature': Field_relatedGenericFeature,
    'field:relatedDiscriminatedFeatureList': Field_relatedDiscriminatedFeatureList,
    'field:relatedDiscriminatedFeature': Field_relatedDiscriminatedFeature,

    'validator:string': Validator_string,
    'validator:integer': Validator_integer,
    'validator:float': Validator_float,
    'validator:geometryConstraint': Validator_geometryConstraint,

    'model:db': DbModel,

}


def register_class(category, name, cls):
    _TYPES[category + ':' + name] = cls


def get_class(category, name):
    cls = _TYPES.get(category + ':' + name)
    if not cls:
        raise gws.Error(f'class not found: {category}:{name}')
    return cls

def clone_mc(mc, **kwargs):
    mc2 = ModelContext(**vars(mc))
    mc2.__dict__.update(kwargs)
    return mc2
