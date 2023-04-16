"""Select widget."""

import gws
import gws.base.model.widget
import gws.types as t

gws.ext.new.modelWidget('select')


# see also js/ui/select
class ListItem(gws.Data):
    value: t.Any
    text: str
    extraText: t.Optional[str]
    level: t.Optional[int]


class Config(gws.base.model.widget.Config):
    items: list[ListItem]


class Props(gws.base.model.widget.Props):
    items: list[ListItem]


class Object(gws.base.model.widget.Object):

    def props(self, user):
        return gws.merge(super().props(user), items=self.cfg('items'))
