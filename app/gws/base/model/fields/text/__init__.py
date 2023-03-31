"""Text field."""

import sqlalchemy as sa

import gws
import gws.base.database.model
import gws.base.model.field
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
    minLength: int
    caseSensitive: bool


class SearchConfig(gws.Config):
    type: SearchType
    minLength: int = 0
    caseSensitive: bool = False


class Config(gws.base.model.field.Config):
    textSearch: t.Optional[SearchConfig]


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Scalar):
    attributeType = gws.AttributeType.str
    textSearch: t.Optional[Search]

    def configure(self):
        self.textSearch = None
        p = self.cfg('textSearch')
        if p:
            self.textSearch = Search(
                type=p.get('type', SearchType.exact),
                minLength=p.get('minLength', 0),
                caseSensitive=p.get('caseSensitive', False),
            )

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
        fld = sa.sql.cast(
            getattr(mod.orm_class(), self.name),
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
