"""Base model class."""

from typing import Optional, cast

import gws
import gws.base.feature
import gws.lib.shape
import gws.config.util

DEFAULT_UID_NAME = 'uid'
DEFAULT_GEOMETRY_NAME = 'geometry'


class TableViewColumn(gws.Data):
    """Column of the feature table view."""

    name: str
    """Name of the model field shown in the column."""
    width: Optional[int]
    """Column width in pixels."""


class ClientOptions(gws.Data):
    """Client-side options for editing features of the model."""

    keepFormOpen: bool = False
    """Keep the edit form open after saving."""


class Config(gws.ConfigWithAccess):
    """Model configuration"""

    fields: Optional[list[gws.ext.config.modelField]]
    """Model fields."""
    loadingStrategy: Optional[gws.FeatureLoadingStrategy]
    """How the client loads features."""
    exportStrategy: Optional[gws.FeatureExportStrategy]
    """How features are obtained for export."""
    title: str = ''
    """Model title."""
    isEditable: bool = False
    """Features of this model can be edited."""
    withAutoFields: bool = False
    """Add fields for unconfigured source columns in addition to the configured ones."""
    excludeColumns: Optional[list[str]]
    """Source columns to skip when creating fields automatically."""
    withTableView: bool = True
    """Enable the table view of features in the client."""
    tableViewColumns: Optional[list[TableViewColumn]]
    """Columns of the table view."""
    templates: Optional[list[gws.ext.config.template]]
    """Templates for rendering features of this model."""
    sort: Optional[list[gws.SortOptions]]
    """Default sort order of features."""
    clientOptions: Optional[ClientOptions]
    """Client-side options for the model."""


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
    """Base model.

    Provides the model configuration, with fields, uid, geometry, sort order and
    templates, and the props, table view columns, validation and conversion between
    features and props, delegated to the fields. Subclasses call `configure_model` from
    their ``configure``, override the ``configure_*`` steps they need and implement
    access to the source.
    """

    def configure(self):
        self.isEditable = self.cfg('isEditable', default=False)
        self.withTableView = self.cfg('withTableView', default=True)
        self.fields = []
        self.geometryCrs = None
        self.geometryName = ''
        self.geometryType = None
        self.uidName = ''
        self.loadingStrategy = self.cfg('loadingStrategy')
        self.exportStrategy = self.cfg('exportStrategy', default=gws.FeatureExportStrategy.load)
        self.title = self.cfg('title')
        self.clientOptions = self.cfg('clientOptions') or gws.Data()

    def post_configure(self):
        if self.isEditable and not self.uidName:
            raise gws.ConfigurationError(f'no primary key found for editable model {self}')

    def configure_model(self):
        """Run the model configuration steps.

        Calls the provider, sources, fields, uid, geometry, sort and templates
        configuration methods in this order.
        """

        self.configure_provider()
        self.configure_sources()
        self.configure_fields()
        self.configure_uid()
        self.configure_geometry()
        self.configure_sort()
        self.configure_templates()

    def configure_provider(self):
        """Configure the data provider of the model.

        The base implementation does nothing.

        Returns:
            True if a provider was configured.
        """
        return False

    def configure_sources(self):
        """Configure the data sources of the model.

        The base implementation does nothing.

        Returns:
            True if sources were configured.
        """
        return False

    def configure_fields(self):
        """Create the model fields.

        Creates the configured ``fields``. If there are none, or ``withAutoFields`` is set,
        also creates fields for the source columns with `configure_auto_fields`.

        Returns:
            True if any fields were created.
        """
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
        """Create fields for the source columns.

        Creates a field for each column returned by ``describe``, except columns listed in
        ``excludeColumns``, columns that already have a field, and columns of a type
        without a default field type.

        Returns:
            True if the source could be described, False otherwise.
        """
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
        """Set ``uidName`` to the name of the primary key field.

        Keeps an already set ``uidName``. Otherwise a name is set only if exactly one field
        is a primary key.

        Returns:
            True if ``uidName`` is set, None otherwise.
        """
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
        """Set the geometry name, type and CRS from the first geometry field.

        Returns:
            True if a geometry field was found, None otherwise.
        """
        for fld in self.fields:
            if getattr(fld, 'geometryType', None):
                self.geometryName = fld.name
                self.geometryType = getattr(fld, 'geometryType')
                self.geometryCrs = getattr(fld, 'geometryCrs')
                return True

    def configure_sort(self):
        """Set the default sort order.

        Uses the configured ``sort``, otherwise sorts by the uid field, if any.

        Returns:
            True if a sort order was configured, False otherwise.
        """
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
        """Create the configured templates.

        Returns:
            True if any templates were created.
        """
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
        """Return the columns of the table view for a user.

        Uses the configured ``tableViewColumns``, or all fields otherwise. Fields the user
        cannot use and fields without a widget that supports the table view are skipped.

        Args:
            user: The user.

        Returns:
            A list of columns, empty if the table view is disabled.
        """
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

    def get_feature(self, uid, mc):
        features = self.get_features([uid], mc)
        if features:
            return features[0]

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
            category=feature.category or '',
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

        # only provide "uid" and "geometry" attributes for the view props
        props.attributes = a

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
