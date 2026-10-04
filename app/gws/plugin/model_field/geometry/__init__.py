"""Geometry field.

A scalar field for geometries, stored as ``gws.Shape`` objects. The geometry
type and the CRS are taken from the config or, if not configured, from the
database column. A field without a known CRS is a configuration error.

The field supports geometry searches: when a model is searched with a shape
(optionally widened by a tolerance) or with bounds, the field adds an
``ST_Intersects`` condition on its column. Shapes from the client are
transformed to the field CRS. Without a configured widget, the field uses a
``geometry`` widget.

Example::

    fields+ {
        name "geom"
        type "geometry"
        geometryType "polygon"
        crs "EPSG:25832"
    }
"""

from typing import Optional, cast

import gws
import gws.base.database.model
import gws.base.model.scalar_field
import gws.base.shape
import gws.lib.crs
import gws.lib.sa as sa


@gws.ext.config.modelField('geometry')
class Config(gws.base.model.scalar_field.Config):
    """Field for geometries."""

    geometryType: Optional[gws.GeometryType]
    """Geometry type."""
    crs: Optional[gws.CrsName]
    """CRS of the geometries."""


@gws.ext.props.modelField('geometry')
class Props(gws.base.model.scalar_field.Props):
    geometryType: gws.GeometryType


@gws.ext.object.modelField('geometry')
class Object(gws.base.model.scalar_field.Object):
    """Geometry field object."""

    attributeType = gws.AttributeType.geometry
    supportsGeometrySearch = True

    geometryType: gws.GeometryType
    """Geometry type of the field."""
    geometryCrs: gws.Crs
    """CRS of the stored geometries."""

    def configure(self):
        setattr(self, 'geometryType', None)
        setattr(self, 'geometryCrs', None)
        self.configure_geometry_type()
        self.configure_geometry_crs()
        if not self.geometryCrs:
            tab = getattr(self.model, 'tableName', '')
            raise gws.ConfigurationError(f'unknown CRS for {tab!r}.{self.name!r}')

    def configure_geometry_type(self):
        """Set the geometry type from the config or from the database column.

        Returns:
            True if the geometry type was set, None otherwise.
        """
        s = self.cfg('geometryType')
        if s:
            self.geometryType = s
            return True

        col = self.describe()
        if col and col.geometryType:
            self.geometryType = col.geometryType
            return True

    def configure_geometry_crs(self):
        """Set the CRS from the config or from the SRID of the database column.

        Returns:
            True if the CRS was set, None otherwise.
        """
        s = self.cfg('crs')
        if s:
            self.geometryCrs = gws.lib.crs.get(s)
            return True

        col = self.describe()
        if col and col.geometrySrid:
            crs = gws.lib.crs.get(col.geometrySrid)
            if crs:
                self.geometryCrs = crs
                return True

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='geometry')
            return True

    ##

    def props(self, user):
        return gws.u.merge(super().props(user), geometryType=self.geometryType)

    ##

    def before_select(self, mc):
        super().before_select(mc)

        shape = None

        if mc.search.shape:
            shape = mc.search.shape
            if mc.search.tolerance:
                tol_value, tol_unit = mc.search.tolerance
                if tol_unit == gws.Uom.px:
                    tol_value *= mc.search.resolution
                shape = shape.tolerance_polygon(tol_value)
        elif mc.search.bounds:
            shape = gws.base.shape.from_bounds(mc.search.bounds)

        if shape:
            shape = shape.transformed_to(self.geometryCrs)

            model = cast(gws.base.database.model.Object, self.model)
            col = model.column(self.name)

            mc.dbSelect.geometryWhere.append(sa.func.st_intersects(col, sa.cast(shape.to_ewkb_hex(), sa.geo.Geometry())))

    def raw_to_python(self, feature, value, mc):
        # here, value is a geosa WKBElement
        return gws.base.shape.from_wkb_hex(str(value))

    def prop_to_python(self, feature, value, mc):
        shape = self._prop_to_shape(value)
        if shape:
            return shape.transformed_to(self.geometryCrs)
        return gws.ErrorValue

    def python_to_raw(self, feature, value, mc):
        return value.to_ewkb_hex()

    def python_to_prop(self, feature, value, mc):
        return cast(gws.Shape, value).to_props()

    def _prop_to_shape(self, value):
        """Convert a shape, shape props or a dict to a shape, or return None."""
        if isinstance(value, gws.base.shape.Shape):
            return value
        if gws.u.is_data_object(value):
            try:
                return gws.base.shape.from_props(value)
            except gws.Error:
                pass
        if gws.u.is_dict(value):
            try:
                return gws.base.shape.from_dict(value)
            except gws.Error:
                pass
