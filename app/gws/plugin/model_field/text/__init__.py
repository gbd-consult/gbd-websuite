"""Text field."""

import gws
import gws.base.database.model
import gws.base.model.scalar_field
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('text')


class SearchType(t.Enum):
    exact = 'exact'
    begin = 'begin'
    end = 'end'
    any = 'any'
    like = 'like'


class Search(gws.Data):
    type: SearchType
    minLength: int = 0
    caseSensitive: bool = False


class Config(gws.base.model.scalar_field.Config):
    textSearch: t.Optional[Search]


class Props(gws.base.model.scalar_field.Props):
    pass


class Object(gws.base.model.scalar_field.Object):
    attributeType = gws.AttributeType.str
    textSearch: t.Optional[Search]

    def configure(self):
        self.textSearch = self.cfg('textSearch')
        if self.textSearch:
            self.supportsKeywordSearch = True

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, {'type': 'input'})
            return True

    ##

    def select(self, sel, user):
        if not self.textSearch or not sel.search or not sel.search.keyword:
            return

        kw = sel.search.keyword
        so = self.textSearch
        if so.minLength and len(kw) < so.minLength:
            return

        mod = t.cast(gws.base.database.model.Object, self.model)
        fld = sa.cast(
            getattr(mod.record_class(), self.name),
            sa.String)

        if so.type == SearchType.exact:
            sel.keywordWhere.append(fld == kw)
        else:
            kw = _escape_like(kw)
            if so.type == 'any':
                kw = '%' + kw + '%'
            if so.type == 'begin':
                kw = kw + '%'
            if so.type == 'end':
                kw = '%' + kw

            if so.caseSensitive:
                sel.keywordWhere.append(fld.like(kw, escape='\\'))
            else:
                sel.keywordWhere.append(fld.ilike(kw, escape='\\'))


def _escape_like(s, escape='\\'):
    return (
        s
        .replace(escape, escape + escape)
        .replace('%', escape + '%')
        .replace('_', escape + '_'))
