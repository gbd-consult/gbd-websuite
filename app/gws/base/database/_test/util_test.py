import gws
import gws.base.database.util as util
import gws.lib.sa as sa
import sqlalchemy.dialects.postgresql

_col = sa.column('name', sa.String)


def _sql(clause):
    return str(clause.compile(dialect=sqlalchemy.dialects.postgresql.dialect(), compile_kwargs={'literal_binds': True}))


def _tso(typ, **kwargs):
    return gws.TextSearchOptions(type=typ, minLength=0, caseSensitive=False, **kwargs)


def test_empty_value():
    assert util.text_search_clause(_col, None, _tso(gws.TextSearchType.any)) is None
    assert util.text_search_clause(_col, '  ', _tso(gws.TextSearchType.any)) is None


def test_min_length():
    tso = _tso(gws.TextSearchType.any)
    tso.minLength = 3
    assert util.text_search_clause(_col, 'ab', tso) is None
    assert util.text_search_clause(_col, 'abc', tso) is not None


def test_no_options_is_exact():
    assert _sql(util.text_search_clause(_col, ' abc ', None)) == "name = 'abc'"


def test_exact():
    assert _sql(util.text_search_clause(_col, 'abc', _tso(gws.TextSearchType.exact))) == "name = 'abc'"


def test_any_escapes_wildcards():
    sql = _sql(util.text_search_clause(_col, 'a%b', _tso(gws.TextSearchType.any)))
    assert 'ILIKE' in sql
    assert "a/%%b" in sql
    assert "ESCAPE '/'" in sql


def test_case_sensitive():
    tso = _tso(gws.TextSearchType.begin)
    tso.caseSensitive = True
    sql = _sql(util.text_search_clause(_col, 'abc', tso))
    assert 'ILIKE' not in sql
    assert 'LIKE' in sql


def test_like_uses_pattern():
    sql = _sql(util.text_search_clause(_col, 'a%c', _tso(gws.TextSearchType.like)))
    assert sql == "name ILIKE 'a%%c' ESCAPE '\\'"
