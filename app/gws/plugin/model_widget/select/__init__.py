"""Select widget."""

from typing import Optional, Any

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('select')


# see also js/ui/select
class ListItem(gws.Data):
    value: Any
    text: str
    extraText: Optional[str]
    level: Optional[int]


class ListItemConfig(gws.Data):
    value: Any
    text: Optional[str]
    extraText: Optional[str]
    level: Optional[int]


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
