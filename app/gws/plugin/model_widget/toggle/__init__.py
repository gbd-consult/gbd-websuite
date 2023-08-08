"""Toggle input widget."""

import gws
import gws.base.model.widget
import gws.types as t

gws.ext.new.modelWidget('toggle')


class Config(gws.base.model.widget.Config):
    kind: str = 'checkbox'
    """Toggle kind: checkbox, radio"""


class Props(gws.base.model.widget.Props):
    kind: str


class Object(gws.base.model.widget.Object):

    def props(self, user):
        return gws.merge(super().props(user), kind=self.cfg('kind', default='checkbox'))
