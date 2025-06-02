"""Select widget."""

from typing import Optional, Any

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('select')


# see also js/ui/select
class ListItem(gws.Data):
    """Item in the select widget."""

    value: Any
    """Value of the item."""
    text: str
    """Text to display for the item."""
    extraText: Optional[str]
    """Additional text to display for the item."""
    level: Optional[int]
    """Optional level for hierarchical items, used for indentation."""


class ListItemConfig(gws.Data):
    """Configuration for a list item in the select widget."""

    value: Any
    """Value of the item."""
    text: Optional[str]
    """Text to display for the item."""
    extraText: Optional[str]
    """Additional text to display for the item."""
    level: Optional[int]
    """Optional level for hierarchical items, used for indentation."""


class Config(gws.base.model.widget.Config):
    """Select widget configuration."""

    items: list[ListItemConfig]
    """List of items to select from."""
    withSearch: bool = False
    """Whether to show a search input field."""


class Props(gws.base.model.widget.Props):
    items: list[ListItem]
    withSearch: bool


class Object(gws.base.model.widget.Object):
    def props(self, user):
        # fmt: off
        items = [
            ListItem(value=it.value, text=it.text or str(it.value)) 
            for it in self.cfg('items', default=[])
        ]
        # fmt: on
        return gws.u.merge(
            super().props(user),
            items=items,
            withSearch=bool(self.cfg('withSearch')),
        )
