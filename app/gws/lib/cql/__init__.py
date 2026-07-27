"""CQL2 support.

Parses CQL2-Text filter expressions and turns them into database expressions.

Reference:
    - https://docs.ogc.org/is/21-065r2/21-065r2.html

Usage::

    cond = cql.SqlBuilder(table).build(cql.parse("a_int > 10 AND S_INTERSECTS(a_geom, POINT(1 1))"))
    sel = sa.select(table).where(cond)

Parse trees
-----------

`parse` returns a tree of plain lists, where the first element is the node type
and the rest are arguments::

    a_int > 10          ['>', ['NAME', 'a_int'], ['INT', 10]]
    a_int IS NULL       ['IS_NULL', ['NAME', 'a_int']]
    a IN (1, 2)         ['IN', ['NAME', 'a'], ['INT', 1], ['INT', 2]]

Node types are listed in `Node`, operators and other keyword sets in `C`.
Literal nodes carry a python value: `['INT', 10]`, `['DATE', datetime.date(...)]`.
A `NAME` node carries the dot-separated parts of a property name: `a.b` is
`['NAME', 'a', 'b']`.

Function calls come in two flavours. Names the standard knows about (`C.FUNCTIONS`)
are checked for arity and emitted lowercased as `FUNCTION`, everything else is
emitted verbatim as `USER_FUNCTION`::

    S_Intersects(g, h)  ['FUNCTION', 's_intersects', ['NAME', 'g'], ['NAME', 'h']]
    myschema.fn(1)      ['USER_FUNCTION', 'myschema.fn', ['INT', 1]]

Builders
--------

`Builder` walks a tree and dispatches on the node type to a `build_<type>` method,
and on the function name to a `func_<name>` method. Missing methods raise `BuildError`,
so a subclass supports exactly what it implements.

`SqlBuilder` generates SQLAlchemy expressions for a postgis table and implements all
standard functions. Subclasses customize single node types, e.g. a model that stores
geometries in a projected crs only overrides the geometry literals::

    class MyBuilder(cql.SqlBuilder):
        def build_wkt(self, args):
            return sa.func.ST_Transform(super().build_wkt(args), 3857)

Backend specific functions are handled by `build_user_function`, which receives the
name as written, followed by the argument nodes.

Notes
-----

- `SqlBuilder` requires postgis, and the `unaccent` extension for the `ACCENTI`
  function (``CREATE EXTENSION unaccent``).
- Temporal predicates compare `tstzrange` values, an instant being a degenerate
  range. Bounds are inclusive, so intervals that only touch do intersect.
  Timestamps without a zone are read as UTC.
- Array literals are accepted both in the standard form ``('a', 'b')`` and as
  ``['a', 'b']``. Arrays are compared as sets, in particular `A_EQUALS` ignores
  order and duplicates.
- `BBOX` is limited to the 2d form with four arguments.
"""

from .parser import parse, ParseError, Node, C
from .builder import Builder, SqlBuilder, BuildError

__all__ = [
    'parse',
    'ParseError',
    'Node',
    'C',
    'Builder',
    'SqlBuilder',
    'BuildError',
]
