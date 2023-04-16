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

    geometryType: gws.GeometryType
    geometryCrs: gws.ICrs

    def configure(self):
        self.supportsGeometrySearch = True

        p = self._get_type_and_crs()
        if p and p[0] and p[1]:
            self.geometryType = p[0]
            self.geometryCrs = p[1]
        else:
            raise gws.Error(f'unable to find type/crs for column {self.name!r}')

    def _get_type_and_crs(self):
        p = self.cfg('geometryType')
        if p:
            return self.cfg('geometryType'), gws.gis.crs.get(self.cfg('crs'))
        desc = self.model.describe()
        if not desc:
            return
        col = desc.columns.get(self.name)
        if not col:
            return
        return col.geometryType, gws.gis.crs.get(col.geometrySrid)

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, {'type': 'geometry'})
            return True

    ##

    def props(self, user):
        return gws.merge(super().props(user), geometryType=self.geometryType)

    ##

    def select(self, sel, user):
        shape = None
        if sel.search.shape:
            shape = sel.search.shape
            if sel.search.tolerance:
                tol_value, tol_unit = sel.search.tolerance
                if tol_unit == gws.Uom.px:
                    tol_value *= sel.search.resolution
                shape = shape.tolerance_polygon(tol_value)
        elif sel.search.bounds:
            shape = gws.base.shape.from_bounds(sel.search.bounds)

        if shape:
            shape = shape.transformed_to(self.geometryCrs)
            mod = t.cast(gws.base.database.model.Object, self.model)
            fld = sa.sql.cast(
                getattr(mod.record_class(), self.name),
                sa.geo.Geometry)
            sel.geometryWhere.append(sa.func.st_intersects(
                fld,
                sa.cast(shape.to_ewkb_hex(), sa.geo.Geometry())))

    def db_to_py(self, val):
        # here, val is a geosa WKBElement
        return gws.base.shape.from_wkb_hex(str(val))

    def prop_to_py(self, val):
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
        return gws.ErrorValue

    def py_to_db(self, val):
        return val.to_ewkb_hex()
