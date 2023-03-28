"""Geometry field."""

import sqlalchemy as sa
import geoalchemy2 as geosa

import gws
import gws.base.database.sql as sql
import gws.base.database.model
import gws.base.shape
import gws.gis.crs
import gws.types as t

import gws.base.model.field
import gws.base.model.fields.scalar as scalar

gws.ext.new.modelField('geometry')


class Config(scalar.Config):
    geometryType: t.Optional[gws.GeometryType]
    crs: t.Optional[gws.CrsName]


class Props(scalar.Props):
    pass


class Object(scalar.Object):
    attributeType = gws.AttributeType.geometry
    geometryType = ''
    geometryCrs: gws.ICrs

    def configure(self):
        self.geometryType = self.var('geometryType')

        self.geometryCrs = None  # type:ignore
        if self.var('crs'):
            self.geometryCrs = gws.gis.crs.get(self.var('crs'))

    def sa_select(self, sel: sql.SelectStatement, user):
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
                getattr(mod.orm_class(), self.name),
                geosa.Geometry)
            sel.geometryWhere.append(sa.func.st_intersects(
                fld,
                sa.cast(shape.to_ewkb_hex(), geosa.Geometry())))

    def load_from_record(self, feature, rec, user):
        if not hasattr(rec, self.name):
            return
        val = getattr(rec, self.name)
        if val is not None:
            feature.attributes[self.name] = gws.base.shape.from_wkb_hex(str(val))

    def load_from_props(self, feature, props, user, **kwargs):
        if self.name not in props.attributes:
            return
        val = props.attributes.get(self.name)
        if isinstance(val, gws.base.shape.Shape):
            feature.attributes[self.name] = val
            return
        if gws.is_data_object(val):
            try:
                feature.attributes[self.name] = gws.base.shape.from_props(val)
                return
            except gws.Error:
                pass
        if gws.is_dict(val):
            try:
                feature.attributes[self.name] = gws.base.shape.from_dict(val)
                return
            except gws.Error:
                pass
        feature.attributes[self.name] = gws.ErrorValue

    def store_to_record(self, feature, rec, user):
        ok, val = self.value_from_feature(feature)
        if ok:
            setattr(rec, self.name, t.cast(gws.base.shape.Shape, val).to_ewkb_hex())
