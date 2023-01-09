"""Geometry field."""

import gws
import gws.base.database.sql as sql
import gws.base.database.model
import gws.base.shape
import gws.gis.crs
import gws.types as t

from .. import scalar


@gws.ext.config.modelField('geometry')
class Config(scalar.Config):
    pass


@gws.ext.props.modelField('geometry')
class Props(scalar.Props):
    pass


@gws.ext.object.modelField('geometry')
class Object(scalar.Object):
    attributeType = gws.AttributeType.geometry
    geometryType: gws.GeometryType
    geometryCrs: gws.ICrs

    def configure(self):
        if hasattr(self.model, 'tableName'):
            desc = self.model.provider.describe_table(self.model.tableName)
            if self.name in desc.columns:
                self.geometryType = desc.columns[self.name].geometryType
                self.geometryCrs = gws.gis.crs.get(desc.columns[self.name].geometrySrid)

    def sa_select(self, sel: sql.SelectStatement, user):
        shape = None
        if sel.search.shape:
            shape = sel.search.shape
            if sel.search.tolerance:
                tol_value, tol_unit = sel.search.tolerance
                if tol_unit == gws.Uom.PX:
                    tol_value *= sel.search.resolution
                shape = shape.tolerance_polygon(tol_value)
        elif sel.search.bounds:
            shape = gws.base.shape.from_bounds(sel.search.bounds)

        if shape:
            shape = shape.transformed_to(self.geometryCrs)
            mod = t.cast(gws.base.database.model.Object, self.model)
            fld = sql.sa.sql.cast(
                getattr(mod.sa_class(), self.name),
                sql.geosa.Geometry)
            sel.geometryWhere.append(sql.sa.func.st_intersects(
                fld,
                sql.sa.cast(shape.to_ewkb_hex(), sql.geosa.Geometry())))

    def load_from_dict(self, feature, attributes, user, **kwargs):
        val = attributes.get(self.name)
        if isinstance(val, gws.base.shape.Shape):
            feature.attributes[self.name] = val
        elif gws.is_data_object(val):
            feature.attributes[self.name] = gws.base.shape.from_props(val)

    def load_from_record(self, feature, record, user, **kwargs):
        val = getattr(record, self.name, None)
        if val:
            feature.attributes[self.name] = gws.base.shape.from_wkb_hex(str(val))

    def store_to_dict(self, feature, attributes, user, **kwargs):
        val = feature.attributes.get(self.name)
        if val:
            attributes[self.name] = gws.props(val, user, self)

    def store_to_record(self, feature, record, user, **kwargs):
        val = feature.attributes.get(self.name)
        if val:
            setattr(record, self.name, t.cast(gws.base.shape.Shape, val).to_ewkb_hex())
