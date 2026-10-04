"""Select widget.

Drop-down list with a fixed set of items, optionally with a search field.
Each item has a value, which is stored in the field, and a text to display,
which defaults to the value. The ``extraText`` and ``level`` options of an
item are not passed to the client.

Example::

    fields+ {
        name "kind"
        type "text"
        widget {
            type "select"
            withSearch true
            items [
                { value "a" text "Type A" }
                { value "b" text "Type B" }
            ]
        }
    }
"""

from typing import Optional, Any

import gws
import gws.base.model.widget


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


class ListItemConfig(gws.Config):
    """Option of a select widget."""

    value: Any
    """Value stored in the field when the item is selected."""
    text: Optional[str]
    """Label shown for the item."""
    extraText: Optional[str]
    """Additional text to display for the item."""
    level: Optional[int]
    """Optional level for hierarchical items, used for indentation."""


@gws.ext.config.modelWidget('select')
class Config(gws.base.model.widget.Config):
    """Drop-down list with a fixed set of values."""

    items: list[ListItemConfig]
    """Options to choose from."""
    withSearch: bool = False
    """Show a search field to filter the options."""


@gws.ext.props.modelWidget('select')
class Props(gws.base.model.widget.Props):
    items: list[ListItem]
    withSearch: bool


@gws.ext.object.modelWidget('select')
class Object(gws.base.model.widget.Object):
    """Select widget object."""

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
