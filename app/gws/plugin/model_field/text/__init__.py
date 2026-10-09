"""Text field.

A scalar field for strings. Without a configured widget, the field uses an
``input`` widget.

With ``textSearch``, the field takes part in keyword searches of database
models. The keyword is compared with the column value cast to a string:

- ``exact``: the value equals the keyword,
- ``any``, ``begin``, ``end``: the keyword occurs anywhere, at the start or at
  the end of the value; ``%`` and ``_`` in the keyword are matched literally,
- ``like``: the keyword is used as an SQL ``LIKE`` pattern as is.

The pattern match is case-insensitive (``ILIKE``) unless ``caseSensitive`` is set.
Keywords shorter than ``minLength`` are ignored.

Example::

    fields+ {
        name "name"
        type "text"
        textSearch { type "begin" minLength 2 }
    }
"""

from typing import Optional, cast

import gws
import gws.base.database.model
import gws.base.database.util
import gws.base.model.scalar_field
import gws.lib.sa as sa


@gws.ext.config.modelField('text')
class Config(gws.base.model.scalar_field.Config):
    """Field for text values."""

    textSearch: Optional[gws.TextSearchOptions]
    """Keyword search options for the field."""


@gws.ext.props.modelField('text')
class Props(gws.base.model.scalar_field.Props):
    pass


@gws.ext.object.modelField('text')
class Object(gws.base.model.scalar_field.Object):
    """Text field object."""

    attributeType = gws.AttributeType.str
    textSearch: Optional[gws.TextSearchOptions]
    """Keyword search options, or None if the field is not searchable."""

    def configure(self):
        self.textSearch = self.cfg('textSearch')
        if self.textSearch:
            self.supportsKeywordSearch = True

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='input')
            return True

    ##

    def before_select(self, mc):
        super().before_select(mc)

        if not self.textSearch:
            return

        model = cast(gws.base.database.model.Object, self.model)
        col = sa.cast(model.column(self.name), sa.String)

        cond = gws.base.database.util.text_search_clause(col, mc.search.keyword, self.textSearch)
        if cond is not None:
            mc.dbSelect.keywordWhere.append(cond)
