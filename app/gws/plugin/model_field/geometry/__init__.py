"""Geometry field."""

import gws
import gws.base.database.model
import gws.base.model.scalar_field
import gws.base.shape
import gws.gis.crs
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('geometry')


class Config(gws.base.model.scalar_field.Config):
    geometryType: t.Optional[gws.GeometryType]
    crs: t.Optional[gws.CrsName]


class Props(gws.base.model.scalar_field.Props):
    geometryType: gws.GeometryType


class Object(gws.base.model.scalar_field.Object):
    attributeType = gws.AttributeType.geometry
    supportsGeometrySearch = True

    geometryType: gws.GeometryType
    geometryCrs: gws.ICrs

    def configure(self):
        setattr(self, 'geometryType', None)
        setattr(self, 'geometryCrs', None)
        self.configure_geometry_type()
        self.configure_geometry_crs()

    def configure_geometry_type(self):
        s = self.cfg('geometryType')
        if s:
            self.geometryType = s
            return True
        c = self.model.describe().columnMap.get(self.name)
        if c and c.geometryType:
            self.geometryType = c.geometryType
            return True

    def configure_geometry_crs(self):
        s = self.cfg('geometryCrs')
        if s:
            self.geometryCrs = gws.gis.crs.get(s)
            return True
        c = self.model.describe().columnMap.get(self.name)
        if c and c.geometrySrid:
            self.geometryCrs = gws.gis.crs.get(c.geometrySrid)
            return True

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='geometry')
            return True

    ##

    def props(self, user):
        return gws.merge(super().props(user), geometryType=self.geometryType)

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

            model = t.cast(gws.base.database.model.Object, self.model)
            col = model.column(self.name)

            mc.dbSelect.geometryWhere.append(sa.func.st_intersects(
                col,
                sa.cast(shape.to_ewkb_hex(), sa.geo.Geometry())))

    def raw_to_python(self, val, mc):
        # here, val is a geosa WKBElement
        return gws.base.shape.from_wkb_hex(str(val))

    def prop_to_python(self, val, mc):
        shape = self._prop_to_shape(val)
        if shape:
            return shape.transformed_to(self.geometryCrs)
        return gws.ErrorValue

    def python_to_raw(self, val, mc):
        return val.to_ewkb_hex()

    def python_to_prop(self, val, mc):
        return t.cast(gws.IShape, val).to_props()

    def _prop_to_shape(self, val):
        if isinstance(val, gws.base.shape.Shape):
            return val
        if gws.is_data_object(val):
            try:
                return gws.base.shape.from_props(val)
            except gws.Error:
                pass
        if gws.is_dict(val):
            try:
                return gws.base.shape.from_dict(val)
            except gws.Error:
                pass

