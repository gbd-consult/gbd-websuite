"""Generic scalar field."""

import gws
import gws.base.database.sql as sql
import gws.base.model


class Config(gws.base.model.field.Config):
    pass


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    def sa_columns(self, cdict):
        kwargs = {}
        if self.isPrimaryKey:
            kwargs['primary_key'] = True
        # if self.value.serverDefault:
        #     kwargs['server_default'] = sa.text(self.value.serverDefault)
        col = sql.sa.Column(self.name, sql.ATTR_TO_SA[self.attributeType], **kwargs)
        cdict[self.name] = col
