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


class ListItemConfig(gws.Data):
    value: t.Any
    text: t.Optional[str]
    extraText: t.Optional[str]
    level: t.Optional[int]


class Config(gws.base.model.widget.Config):
    items: list[ListItemConfig]
    withSearch: bool = False


class Props(gws.base.model.widget.Props):
    items: list[ListItem]
    withSearch: bool


class Object(gws.base.model.widget.Object):
    def props(self, user):
        items = [
            ListItem(value=it.value, text=it.text or str(it.value))
            for it in self.cfg('items', default=[])
        ]
        return gws.u.merge(
            super().props(user),
            items=items,
            withSearch=bool(self.cfg('withSearch'))
        )
