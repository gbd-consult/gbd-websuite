"""Base model."""

from typing import Optional, cast

import gws
import gws.base.feature
import gws.base.shape
import gws.config.util

DEFAULT_UID_NAME = 'uid'
DEFAULT_GEOMETRY_NAME = 'geometry'


class TableViewColumn(gws.Data):
    """Table view column configuration."""

    name: str
    """Column name."""
    width: Optional[int]
    """Column width in pixels. If not set, the column will be auto-sized."""


class ClientOptions(gws.Data):
    """Client options for a model"""

    keepFormOpen: bool = False
    """Keep the edit form open after save"""


class Config(gws.ConfigWithAccess):
    """Model configuration"""

    fields: Optional[list[gws.ext.config.modelField]]
    """Model fields."""
    loadingStrategy: Optional[gws.FeatureLoadingStrategy]
    """Loading strategy for features."""
    title: str = ''
    """Model title."""
    isEditable: bool = False
    """This model is editable."""
    withAutoFields: bool = False
    """Autoload non-configured model fields from the source."""
    excludeColumns: Optional[list[str]]
    """Exclude columns names from autoload."""
    withTableView: bool = True
    """Enable table view for this model."""
    tableViewColumns: Optional[list[TableViewColumn]]
    """Fields to include in the table view."""
    templates: Optional[list[gws.ext.config.template]]
    """Feature templates."""
    sort: Optional[list[gws.SortOptions]]
    """Default sorting."""
    clientOptions: Optional[ClientOptions]
    """Client options for a model. (added in 8.1)."""


class Props(gws.Props):
    clientOptions: gws.ModelClientOptions
    canCreate: bool
    canDelete: bool
    canRead: bool
    canWrite: bool
    isEditable: bool
    fields: list[gws.ext.props.modelField]
    geometryCrs: Optional[str]
    geometryName: Optional[str]
    geometryType: Optional[gws.GeometryType]
    layerUid: Optional[str]
    loadingStrategy: gws.FeatureLoadingStrategy
    supportsGeometrySearch: bool
    supportsKeywordSearch: bool
    tableViewColumns: list[TableViewColumn]
    title: str
    uid: str
    uidName: Optional[str]


class Object(gws.Model):
    def configure(self):
        self.isEditable = self.cfg('isEditable', default=False)
        self.withTableView = self.cfg('withTableView', default=True)
        self.fields = []
        self.geometryCrs = None
        self.geometryName = ''
        self.geometryType = None
        self.uidName = ''
        self.loadingStrategy = self.cfg('loadingStrategy')
        self.title = self.cfg('title')
        self.clientOptions = self.cfg('clientOptions') or gws.Data()

    def post_configure(self):
        if self.isEditable and not self.uidName:
            raise gws.ConfigurationError(f'no primary key found for editable model {self}')

    def configure_model(self):
        """Model configuration protocol."""

        self.configure_provider()
        self.configure_sources()
        self.configure_fields()
        self.configure_uid()
        self.configure_geometry()
        self.configure_sort()
        self.configure_templates()

    def configure_provider(self):
        return False

    def configure_sources(self):
        return False

    def configure_fields(self):
        has_conf = False
        has_auto = False

        p = self.cfg('fields')
        if p:
            self.fields = self.create_children(gws.ext.object.modelField, p, _defaultModel=self)
            has_conf = True
        if not has_conf or self.cfg('withAutoFields'):
            has_auto = self.configure_auto_fields()

        return has_conf or has_auto

    def configure_auto_fields(self):
        desc = self.describe()
        if not desc:
            return False

        exclude = set(self.cfg('excludeColumns', default=[]))
        exclude.update(fld.name for fld in self.fields)

        for col in desc.columns:
            if col.name in exclude:
                continue

            typ = _DEFAULT_FIELD_TYPES.get(col.type)
            if not typ:
                # gws.log.warning(f'cannot find suitable field type for column {desc.fullName}.{col.name} ({col.type})')
                continue

            cfg = gws.Config(
                type=typ,
                name=col.name,
                isPrimaryKey=col.isPrimaryKey,
                isRequired=not col.isNullable,
            )
            fld = self.create_child(gws.ext.object.modelField, cfg, _defaultModel=self)
            if fld:
                self.fields.append(fld)
                exclude.add(fld.name)

        return True

    def configure_uid(self):
        if self.uidName:
            return True
        uids = []
        for fld in self.fields:
            if fld.isPrimaryKey:
                uids.append(fld.name)
        if len(uids) == 1:
            self.uidName = uids[0]
            return True

    def configure_geometry(self):
        for fld in self.fields:
            if getattr(fld, 'geometryType', None):
                self.geometryName = fld.name
                self.geometryType = getattr(fld, 'geometryType')
                self.geometryCrs = getattr(fld, 'geometryCrs')
                return True

    def configure_sort(self):
        p = self.cfg('sort')
        if p:
            self.defaultSort = [gws.SearchSort(c) for c in p]
            return True
        if self.uidName:
            self.defaultSort = [gws.SearchSort(fieldName=self.uidName, reversed=False)]
            return False
        self.defaultSort = []
        return False

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self)

    ##

    def props(self, user):
        layer = cast(gws.Layer, self.find_closest(gws.ext.object.layer))

        return gws.Props(
            clientOptions=self.clientOptions,
            canCreate=user.can_create(self),
            canDelete=user.can_delete(self),
            canRead=user.can_read(self),
            canWrite=user.can_write(self),
            fields=self.fields,
            geometryCrs=self.geometryCrs.epsg if self.geometryCrs else None,
            geometryName=self.geometryName,
            geometryType=self.geometryType,
            isEditable=self.isEditable,
            layerUid=layer.uid if layer else None,
            loadingStrategy=self.loadingStrategy or (layer.loadingStrategy if layer else gws.FeatureLoadingStrategy.all),
            supportsGeometrySearch=any(fld.supportsGeometrySearch for fld in self.fields),
            supportsKeywordSearch=any(fld.supportsKeywordSearch for fld in self.fields),
            tableViewColumns=self.table_view_columns(user),
            title=self.title or (layer.title if layer else ''),
            uid=self.uid,
            uidName=self.uidName,
        )

    ##

    def table_view_columns(self, user):
        if not self.withTableView:
            return []

        cols = []

        p = self.cfg('tableViewColumns')
        if p:
            fmap = {fld.name: fld for fld in self.fields}
            for c in p:
                fld = fmap.get(c.name)
                if fld and user.can_use(fld) and fld.widget and fld.widget.supportsTableView:
                    cols.append(TableViewColumn(name=c.name, width=c.width or 0))
        else:
            for fld in self.fields:
                if fld and user.can_use(fld) and fld.widget and fld.widget.supportsTableView:
                    cols.append(TableViewColumn(name=fld.name, width=0))

        return cols

    def field(self, name):
        for fld in self.fields:
            if fld.name == name:
                return fld

    def validate_feature(self, feature, mc):
        feature.errors = []
        for fld in self.fields:
            fld.do_validate(feature, mc)
        return len(feature.errors) == 0

    def related_models(self):
        d = {}

        for fld in self.fields:
            for model in fld.related_models():
                d[model.uid] = model

        return list(d.values())

    ##

    def get_features(self, uids, mc):
        if not uids:
            return []
        search = gws.SearchQuery(uids=set(uids))
        return self.find_features(search, mc)

    def find_features(self, search, mc):
        return []

    ##

    def feature_from_props(self, props, mc):
        props = cast(gws.FeatureProps, gws.u.to_data_object(props))
        feature = gws.base.feature.new(model=self, props=props)
        feature.cssSelector = props.cssSelector or ''
        feature.isNew = props.isNew or False
        feature.views = props.views or {}

        for fld in self.fields:
            fld.from_props(feature, mc)

        return feature

    def feature_to_props(self, feature, mc):
        feature.props = gws.FeatureProps(
            attributes={},
            cssSelector=feature.cssSelector,
            errors=feature.errors or [],
            isNew=feature.isNew,
            modelUid=self.uid,
            uid=feature.uid(),
            views=feature.views,
        )

        for fld in self.fields:
            fld.to_props(feature, mc)

        return feature.props

    def feature_to_view_props(self, feature, mc):
        props = self.feature_to_props(feature, mc)

        a = {}

        if self.uidName:
            a[DEFAULT_UID_NAME] = props.attributes.get(self.uidName)
        if self.geometryName:
            a[DEFAULT_GEOMETRY_NAME] = props.attributes.get(self.geometryName)

        props.attributes = a
        props.modelUid = ''
        props.category = feature.category or ''

        return props


##

# @TODO this should be populated dynamically from available gws.ext.object.modelField types

_DEFAULT_FIELD_TYPES = {
    gws.AttributeType.str: 'text',
    gws.AttributeType.int: 'integer',
    gws.AttributeType.date: 'date',
    gws.AttributeType.bool: 'bool',
    # gws.AttributeType.bytes: 'bytea',
    gws.AttributeType.datetime: 'datetime',
    # gws.AttributeType.feature: 'feature',
    # gws.AttributeType.featurelist: 'featurelist',
    gws.AttributeType.float: 'float',
    # gws.AttributeType.floatlist: 'floatlist',
    gws.AttributeType.geometry: 'geometry',
    # gws.AttributeType.intlist: 'intlist',
    # gws.AttributeType.strlist: 'strlist',
    gws.AttributeType.time: 'time',
}
