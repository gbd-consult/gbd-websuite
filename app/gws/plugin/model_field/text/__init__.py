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
            self.widget = self.create_child(gws.ext.object.modelWidget, type='input')
            return True

    ##

    def augment_select(self, sel, user):
        if not self.textSearch or not sel.search or not sel.search.keyword:
            return

        kw = sel.search.keyword
        ts = self.textSearch
        if ts.minLength and len(kw) < ts.minLength:
            return

        mod = t.cast(gws.base.database.model.Object, self.model)
        fld = sa.cast(
            getattr(mod.record_class(), self.name),
            sa.String)

        if ts.type == gws.TextSearchType.exact:
            sel.keywordWhere.append(fld == kw)
        else:
            if ts.type == gws.TextSearchType.any:
                kw = '%' + _escape_like(kw) + '%'
            if ts.type == gws.TextSearchType.begin:
                kw = _escape_like(kw) + '%'
            if ts.type == gws.TextSearchType.end:
                kw = '%' + _escape_like(kw)
            if ts.type == gws.TextSearchType.like:
                pass

            if ts.caseSensitive:
                sel.keywordWhere.append(fld.like(kw, escape='\\'))
            else:
                sel.keywordWhere.append(fld.ilike(kw, escape='\\'))


def _escape_like(s, escape='\\'):
    return (
        s
        .replace(escape, escape + escape)
        .replace('%', escape + '%')
        .replace('_', escape + '_'))
