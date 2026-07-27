"""Build database expressions from CQL2 parse trees.

`Builder` only dispatches on node types, `SqlBuilder` generates SQLAlchemy
expressions for postgis. See the package documentation for details.
"""

from typing import Any, cast

import operator

import gws
import gws.lib.crs
import gws.lib.datetimex as dtx
import gws.lib.sa as sa

from .parser import Node, C


class BuildError(Exception):
    pass


class Builder:
    def get_method(self, name):
        """Return a handler method, or `None` if the subclass doesn't implement it."""

        return getattr(self, name.lower(), None)

    def build(self, e):
        """Build an expression from a parse tree node."""

        fn = self.get_method('build_' + e[0])
        if fn:
            return fn(e[1:])

        if e[0] in C.OPERATORS:
            return self.build_operator(e[0], e[1:])

        raise BuildError(f'CQL: node {e[0]!r} not implemented')

    def build_operator(self, op, args):
        """Build a binary operator expression."""

        raise BuildError(f'CQL: operator {op!r} not implemented')

    def build_function(self, args):
        """Build a standard function call, dispatching on the function name."""

        # [FUNCTION, name, arg1, arg2, ...]

        fn = self.get_method('func_' + args[0])
        if fn:
            return fn(args[1:])

        raise BuildError(f'CQL: function {args[0]!r} not implemented')

    def build_user_function(self, args):
        """Build a non-standard function call. Subclasses handle their own functions here."""

        # [USER_FUNCTION, name, arg1, arg2, ...]

        raise BuildError(f'CQL: function {args[0]!r} not implemented')

    def value(self, e) -> Any:
        """Unwrap a literal node into a plain python value."""

        if e[0] == Node.ARRAY:
            return [self.value(a) for a in e[1:]]
        if e[0] in C.LITERALS:
            return e[1]
        raise BuildError(f'CQL: expected a literal, got {e[0]!r}')


class SqlBuilder(Builder):
    _binary_ops = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '=': operator.eq,
        '!=': operator.ne,
        '<>': operator.ne,
        '*': operator.mul,
        '/': operator.truediv,
        '+': operator.add,
        '-': operator.sub,
        '%': operator.mod,
        # sqlalchemy columns don't support '**'
        '^': sa.func.power,
    }

    def __init__(self, table: sa.Table):
        self.table = table

    def build_operator(self, op, args):
        fn = self._binary_ops.get(op)
        if fn:
            a, b = args
            return fn(self.build(a), self.build(b))

        return super().build_operator(op, args)

    def build_name(self, args):
        col = self.table.c.get(args[0])
        if col is None:
            raise BuildError(f'CQL: unknown column {args[0]!r}')
        return col

    def build_array(self, args):
        return [self.build(a) for a in args]

    def build_bool(self, args):
        return self.literal(args[0])

    def build_float(self, args):
        return self.literal(args[0])

    def build_int(self, args):
        return self.literal(args[0])

    def build_string(self, args):
        return self.literal(args[0])

    def build_date(self, args):
        return sa.cast(args[0], sa.DATE())

    def build_timestamp(self, args):
        return sa.cast(args[0], sa.TIMESTAMP(timezone=True))

    def build_wkt(self, args):
        return sa.func.ST_GeomFromText(args[0], gws.lib.crs.WGS84.srid)

    def build_bbox(self, args):
        minx, miny, maxx, maxy = args
        return sa.func.ST_MakeEnvelope(minx, miny, maxx, maxy, gws.lib.crs.WGS84.srid)

    ##

    def build_and(self, args):
        return sa.and_(*[cast(sa.BinaryExpression, self.build(a)) for a in args])

    def build_or(self, args):
        return sa.or_(*[cast(sa.BinaryExpression, self.build(a)) for a in args])

    def build_not(self, args):
        return sa.not_(cast(sa.BinaryExpression, self.build(args[0])))

    def build_between(self, args):
        col = self.build(args[0])
        a = self.build(args[1])
        b = self.build(args[2])
        return col.between(a, b)

    def build_not_between(self, args):
        return sa.not_(self.build_between(args))

    def build_in(self, args):
        col = self.build(args[0])
        ls = [self.build(a) for a in args[1:]]
        return col.in_(ls)

    def build_not_in(self, args):
        return sa.not_(self.build_in(args))

    def build_like(self, args):
        col = self.build(args[0])
        return col.like(self.build(args[1]))

    def build_not_like(self, args):
        return sa.not_(self.build_like(args))

    def build_is_null(self, args):
        col = self.build(args[0])
        return col.is_(None)

    def build_not_null(self, args):
        col = self.build(args[0])
        return col.isnot(None)

    ##

    def func_s_intersects(self, args):
        return sa.func.ST_Intersects(self.build(args[0]), self.build(args[1]))

    def func_s_contains(self, args):
        return sa.func.ST_Contains(self.build(args[0]), self.build(args[1]))

    def func_s_crosses(self, args):
        return sa.func.ST_Crosses(self.build(args[0]), self.build(args[1]))

    def func_s_disjoint(self, args):
        return sa.func.ST_Disjoint(self.build(args[0]), self.build(args[1]))

    def func_s_equals(self, args):
        return sa.func.ST_Equals(self.build(args[0]), self.build(args[1]))

    def func_s_overlaps(self, args):
        return sa.func.ST_Overlaps(self.build(args[0]), self.build(args[1]))

    def func_s_touches(self, args):
        return sa.func.ST_Touches(self.build(args[0]), self.build(args[1]))

    def func_s_within(self, args):
        return sa.func.ST_Within(self.build(args[0]), self.build(args[1]))

    ##

    def func_casei(self, args):
        return sa.func.lower(self.build(args[0]))

    def func_accenti(self, args):
        return sa.func.unaccent(self.build(args[0]))

    ##

    def func_bbox(self, args):
        return self.build_bbox([self.value(a) for a in args])

    def func_timestamp(self, args):
        dt = dtx.from_iso_string(self.value(args[0]), 'UTC')
        return sa.cast(dt, sa.TIMESTAMP(timezone=True))

    def func_date(self, args):
        dt = dtx.from_iso_string(self.value(args[0]), 'UTC')
        return sa.cast(dt, sa.DATE())

    def func_interval(self, args):
        raise BuildError('CQL: INTERVAL is only allowed in temporal predicates')

    ##

    def func_t_equals(self, args):
        a, b = self.temporal_pair(args)
        return a == b

    def func_t_after(self, args):
        a, b = self.temporal_pair(args)
        return sa.func.lower(a) > sa.func.upper(b)

    def func_t_before(self, args):
        a, b = self.temporal_pair(args)
        return sa.func.upper(a) < sa.func.lower(b)

    def func_t_meets(self, args):
        a, b = self.temporal_pair(args)
        return sa.func.upper(a) == sa.func.lower(b)

    def func_t_metby(self, args):
        a, b = self.temporal_pair(args)
        return sa.func.lower(a) == sa.func.upper(b)

    def func_t_during(self, args):
        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) > sa.func.lower(b),
            sa.func.upper(a) < sa.func.upper(b),
        )

    def func_t_contains(self, args):
        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) < sa.func.lower(b),
            sa.func.upper(a) > sa.func.upper(b),
        )

    def func_t_overlaps(self, args):
        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) < sa.func.lower(b),
            sa.func.upper(a) > sa.func.lower(b),
            sa.func.upper(a) < sa.func.upper(b),
        )

    def func_t_overlappedby(self, args):
        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) > sa.func.lower(b),
            sa.func.lower(a) < sa.func.upper(b),
            sa.func.upper(a) > sa.func.upper(b),
        )

    def func_t_starts(self, args):
        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) == sa.func.lower(b),
            sa.func.upper(a) < sa.func.upper(b),
        )

    def func_t_startedby(self, args):
        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) == sa.func.lower(b),
            sa.func.upper(a) > sa.func.upper(b),
        )

    def func_t_finishes(self, args):
        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.upper(a) == sa.func.upper(b),
            sa.func.lower(a) > sa.func.lower(b),
        )

    def func_t_finishedby(self, args):
        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.upper(a) == sa.func.upper(b),
            sa.func.lower(a) < sa.func.lower(b),
        )

    def func_t_intersects(self, args):
        a, b = self.temporal_pair(args)
        return a.op('&&')(b)

    def func_t_disjoint(self, args):
        a, b = self.temporal_pair(args)
        return sa.not_(a.op('&&')(b))

    ##

    def func_a_equals(self, args):
        a, b = self.array_pair(args)
        return sa.and_(a.op('@>')(b), a.op('<@')(b))

    def func_a_contains(self, args):
        a, b = self.array_pair(args)
        return a.op('@>')(b)

    def func_a_containedby(self, args):
        a, b = self.array_pair(args)
        return a.op('<@')(b)

    def func_a_overlaps(self, args):
        a, b = self.array_pair(args)
        return a.op('&&')(b)

    ##

    def temporal_pair(self, args):
        """Coerce both operands of a temporal predicate to ranges."""

        return self.temporal_range(args[0]), self.temporal_range(args[1])

    def temporal_range(self, e):
        """Coerce a temporal expression to a `tstzrange`, an instant becomes a degenerate range.

        A null bound is unbounded in postgres, therefore null inputs must yield a null range,
        otherwise a null column would match everything.
        """

        if e[0] == Node.FUNCTION and e[1] == 'interval':
            lo = self.temporal_bound(e[2], '-infinity')
            hi = self.temporal_bound(e[3], 'infinity')
        else:
            lo = hi = self.timestamp_value(e)

        return sa.case(
            (sa.or_(lo.is_(None), hi.is_(None)), sa.null()),
            else_=sa.func.tstzrange(lo, hi, '[]'),
        )

    def temporal_bound(self, e, unbounded):
        """Build an interval bound, the string `'..'` meaning open."""

        if e[0] == Node.STRING and e[1] == '..':
            return sa.cast(sa.literal(unbounded), sa.TIMESTAMP(timezone=True))
        return self.timestamp_value(e)

    def timestamp_value(self, e):
        """Coerce an expression to a `timestamptz`, naive values are assumed to be UTC."""

        x = self.build(e)
        typ = getattr(x, 'type', None)
        if isinstance(typ, sa.TIMESTAMP) and typ.timezone:
            return x
        return sa.cast(sa.func.timezone('UTC', sa.cast(x, sa.TIMESTAMP())), sa.TIMESTAMP(timezone=True))

    ##

    def literal(self, val):
        """Wrap a python value as a bound parameter."""

        return sa.literal(val)

    ##

    def array_pair(self, args):
        """Coerce both operands of an array predicate to arrays."""

        return self.array_operand(args[0]), self.array_operand(args[1])

    def array_operand(self, e):
        """Build an array expression, an array literal becoming a typed parameter."""

        if e[0] != Node.ARRAY:
            return self.build(e)
        vals = self.value(e)
        return sa.literal(vals, sa.ARRAY(self.array_element_type(vals)))

    def array_element_type(self, vals):
        """Infer the element type of an array literal from its first element."""

        if not vals:
            return sa.Text()
        v = vals[0]
        if isinstance(v, bool):
            return sa.Boolean()
        if isinstance(v, int):
            return sa.Integer()
        if isinstance(v, float):
            return sa.Float()
        return sa.Text()
