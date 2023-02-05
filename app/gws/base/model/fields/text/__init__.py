"""Text field."""

import gws
import gws.base.database.sql as sql
import gws.base.database.model
import gws.types as t

from .. import scalar

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


class Config(scalar.Config):
    textSearch: t.Optional[SearchConfig]


class Props(scalar.Props):
    pass


class Object(scalar.Object):
    attributeType = gws.AttributeType.str
    textSearch: t.Optional[Search]

    def configure(self):
        self.textSearch = None
        p = self.var('textSearch')
        if p:
            self.textSearch = Search(
                type=p.get('type', SearchType.exact),
                minLength=p.get('minLength', 0),
                caseSensitive=p.get('caseSensitive', False),
            )

    def sa_select(self, sel, user):
        sel = t.cast(sql.SelectStatement, sel)

        if not self.textSearch or not sel.search or not sel.search.keyword:
            return

        kw = sel.search.keyword
        so = self.textSearch
        if so.minLength and len(kw) < so.minLength:
            return

        mod = t.cast(gws.base.database.model.Object, self.model)
        fld = sql.sa.sql.cast(
            getattr(mod.sa_class(), self.name),
            sql.sa.String)

        if so.type == SearchType.exact:
            sel.keywordWhere.append(fld == kw)
        else:
            kw = sql.escape_like(kw)
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
