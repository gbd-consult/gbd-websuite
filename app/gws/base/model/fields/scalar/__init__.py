"""Generic scalar field."""

import sqlalchemy as sa
import gws
import gws.base.model.field
import gws.base.database.sql


class Config(gws.base.model.field.Config):
    pass


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    def columns(self):
        kwargs = {}
        if self.isPrimaryKey:
            kwargs['primary_key'] = True
        # if self.value.serverDefault:
        #     kwargs['server_default'] = sa.text(self.value.serverDefault)
        col = sa.Column(self.name, gws.base.database.sql.ATTR_TO_SA[self.attributeType], **kwargs)
        return [col]
