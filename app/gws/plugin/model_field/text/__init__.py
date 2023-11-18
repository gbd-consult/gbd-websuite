"""Text field."""

import gws
import gws.base.database.model
import gws.base.model.scalar_field
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('text')


class Config(gws.base.model.scalar_field.Config):
    textSearch: t.Optional[gws.TextSearchOptions]


class Props(gws.base.model.scalar_field.Props):
    pass


class Object(gws.base.model.scalar_field.Object):
    attributeType = gws.AttributeType.str
    textSearch: t.Optional[gws.TextSearchOptions]

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

        kw = mc.search.keyword
        ts = self.textSearch

        if not kw or not ts or (ts.minLength and len(kw) < ts.minLength):
            return

        model = t.cast(gws.base.database.model.Object, self.model)
        col = sa.cast(model.column(self.name), sa.String)

        if ts.type == gws.TextSearchType.exact:
            mc.dbSelect.keywordWhere.append(col.__eq__(kw))
            return

        if ts.type == gws.TextSearchType.any:
            kw = '%' + _escape_like(kw) + '%'
        elif ts.type == gws.TextSearchType.begin:
            kw = _escape_like(kw) + '%'
        elif ts.type == gws.TextSearchType.end:
            kw = '%' + _escape_like(kw)
        elif ts.type == gws.TextSearchType.like:
            pass

        if ts.caseSensitive:
            mc.dbSelect.keywordWhere.append(col.like(kw, escape='\\'))
        else:
            mc.dbSelect.keywordWhere.append(col.ilike(kw, escape='\\'))


def _escape_like(s, escape='\\'):
    return (
        s
        .replace(escape, escape + escape)
        .replace('%', escape + '%')
        .replace('_', escape + '_'))
