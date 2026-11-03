"""Build database expressions from CQL2 parse trees."""

import datetime
import operator
from typing import Any, cast

import gws
import gws.lib.crs
import gws.lib.datetimex as dtx
import gws.lib.sa as sa

from . import parser
from .parser import C, Node, ParseError


class BuildError(Exception):
    """A parse tree node or function is not supported by the builder, or its arguments are invalid."""


class _Operand(gws.Data):
    """A comparison operand: a built expression, a temporal literal or a pending string."""

    node: list
    isDate: bool
    isTimeStamp: bool
    value: Any
    expr: Any


class Builder:
    """Walks a CQL2 parse tree and dispatches nodes to handler methods.

    Node types go to ``build_<type>`` methods, standard functions to ``func_<name>`` methods
    (names are lowercased). This class implements no node types; subclasses add handlers.
    """

    def get_method(self, name):
        """Find a handler method.

        Args:
            name: Method name, case-insensitive.

        Returns:
            The method, or ``None`` if the builder does not implement it.
        """

        return getattr(self, name.lower(), None)

    def build(self, e):
        """Build an expression from a parse tree node.

        Calls ``build_<type>`` with the node arguments, or ``build_operator`` for operators without one.

        Args:
            e: Parse tree node.

        Returns:
            The expression returned by the handler.

        Raises:
            BuildError: If the node type is not implemented.
        """

        fn = self.get_method('build_' + e[0])
        if fn:
            return fn(e[1:])

        if e[0] in C.OPERATORS:
            return self.build_operator(e[0], e[1:])

        raise BuildError(f'CQL: node {e[0]!r} not implemented')

    def build_operator(self, op, args):
        """Build an operator expression.

        Args:
            op: Operator, e.g. ``>`` or ``+``.
            args: Operand nodes.

        Returns:
            The expression. Not implemented in this class.

        Raises:
            BuildError: Always, in this class.
        """

        raise BuildError(f'CQL: operator {op!r} not implemented')

    def build_function(self, args):
        """Build a standard function call by calling ``func_<name>`` with the argument nodes.

        Args:
            args: Function name, followed by the argument nodes.

        Returns:
            The expression returned by the handler.

        Raises:
            BuildError: If the function is not implemented.
        """

        # [FUNCTION, name, arg1, arg2, ...]

        fn = self.get_method('func_' + args[0])
        if fn:
            return fn(args[1:])

        raise BuildError(f'CQL: function {args[0]!r} not implemented')

    def build_user_function(self, args):
        """Build a non-standard function call. Subclasses handle their own functions here.

        Args:
            args: Function name as written, followed by the argument nodes.

        Returns:
            The expression. Not implemented in this class.

        Raises:
            BuildError: Always, in this class.
        """

        # [USER_FUNCTION, name, arg1, arg2, ...]

        raise BuildError(f'CQL: function {args[0]!r} not implemented')

    def value(self, e) -> Any:
        """Unwrap a literal node into a plain python value.

        Args:
            e: Literal node. Array nodes are unwrapped recursively.

        Returns:
            The value, a list for arrays.

        Raises:
            BuildError: If the node is not a literal.
        """

        if e[0] == Node.ARRAY:
            return [self.value(a) for a in e[1:]]
        if e[0] in C.LITERALS:
            return e[1]
        raise BuildError(f'CQL: expected a literal, got {e[0]!r}')


_BINARY_OPS = {
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

_MIRRORED_OPS = {
    '=': '=',
    '<>': '<>',
    '!=': '!=',
    '<': '>',
    '<=': '>=',
    '>': '<',
    '>=': '<=',
}


class SqlBuilder(Builder):
    """Builds SQLAlchemy expressions for a postgis table.

    Names refer to table columns, literals become bound parameters, geometries are in WGS84.
    All standard CQL2 functions are implemented, user functions are not.

    Dates and timestamps are compared by the date of the timestamp. Strings compared with
    a date or timestamp column are read as dates or timestamps.
    """

    def __init__(self, table: sa.Table, default_time_zone: str = ''):
        """Create a builder.

        Args:
            table: Table whose columns are referenced by names.
            default_time_zone: Time zone for timestamps without a zone and for day boundaries,
                as in ``dtx.time_zone``. The local time zone by default.
        """

        self.table = table
        self.default_time_zone = default_time_zone

    def build_operator(self, op, args):
        """Build a comparison or an arithmetic operator.

        Args:
            op: Operator, e.g. ``>`` or ``+``.
            args: Two operand nodes.

        Returns:
            The expression. Comparisons are built by ``compare``.

        Raises:
            BuildError: If the operator is not implemented.
        """

        if op in C.COMPARISON_OPERATORS:
            a, b = args
            return self.compare(op, self.operand(a), self.operand(b))

        fn = _BINARY_OPS.get(op)
        if fn:
            a, b = args
            return fn(self.build(a), self.build(b))

        return super().build_operator(op, args)

    def build_name(self, args):
        """Build a column reference.

        Args:
            args: Name parts, only the first is used as the column name.

        Returns:
            The table column.

        Raises:
            BuildError: If the table has no such column.
        """

        col = self.table.c.get(args[0])
        if col is None:
            raise BuildError(f'CQL: unknown column {args[0]!r}')
        return col

    def build_neg(self, args):
        """Build a negation.

        Args:
            args: A list with the operand node.

        Returns:
            The negated expression.
        """

        return -self.build(args[0])

    def build_array(self, args):
        """Build an array.

        Args:
            args: Element nodes.

        Returns:
            A list of expressions.
        """

        return [self.build(a) for a in args]

    def build_bool(self, args):
        """Build a boolean literal.

        Args:
            args: A list with the value.

        Returns:
            A bound parameter.
        """

        return self.literal(args[0])

    def build_float(self, args):
        """Build a float literal.

        Args:
            args: A list with the value.

        Returns:
            A bound parameter.
        """

        return self.literal(args[0])

    def build_int(self, args):
        """Build a integer literal.

        Args:
            args: A list with the value.

        Returns:
            A bound parameter.
        """

        return self.literal(args[0])

    def build_string(self, args):
        """Build a string literal.

        Args:
            args: A list with the value.

        Returns:
            A bound parameter.
        """

        return self.literal(args[0])

    def build_wkt(self, args):
        """Build a geometry literal.

        Args:
            args: A list with the WKT string.

        Returns:
            A geometry in WGS84.
        """

        return sa.func.ST_GeomFromText(args[0], gws.lib.crs.WGS84.srid)

    def build_bbox(self, args):
        """Build a bounding box geometry.

        Args:
            args: Four numbers: min x, min y, max x, max y.

        Returns:
            A rectangle geometry in WGS84.
        """

        minx, miny, maxx, maxy = args
        return sa.func.ST_MakeEnvelope(minx, miny, maxx, maxy, gws.lib.crs.WGS84.srid)

    ##

    def build_and(self, args):
        """Build a conjunction.

        Args:
            args: Operand nodes.

        Returns:
            An ``AND`` expression.
        """

        return sa.and_(*[cast(sa.BinaryExpression, self.build(a)) for a in args])

    def build_or(self, args):
        """Build a disjunction.

        Args:
            args: Operand nodes.

        Returns:
            An ``OR`` expression.
        """

        return sa.or_(*[cast(sa.BinaryExpression, self.build(a)) for a in args])

    def build_not(self, args):
        """Build a negation.

        Args:
            args: A list with the operand node.

        Returns:
            A ``NOT`` expression.
        """

        return sa.not_(cast(sa.BinaryExpression, self.build(args[0])))

    def build_between(self, args):
        """Build a ``BETWEEN`` predicate.

        A temporal value is built as ``a >= b AND a <= c``, see ``compare``.

        Args:
            args: The value, lower bound and upper bound nodes.

        Returns:
            A ``BETWEEN`` expression.
        """

        x = self.operand(args[0])
        if x.isDate or x.isTimeStamp:
            return sa.and_(
                self.compare('>=', x, self.operand(args[1])),
                self.compare('<=', x, self.operand(args[2])),
            )
        return self.operand_sql(x).between(self.build(args[1]), self.build(args[2]))

    def build_not_between(self, args):
        """Build a ``NOT BETWEEN`` predicate.

        Args:
            args: The value, lower bound and upper bound nodes.

        Returns:
            A negated ``BETWEEN`` expression.
        """

        return sa.not_(self.build_between(args))

    def build_in(self, args):
        """Build an ``IN`` predicate.

        A temporal value is built as ``a = b OR a = c ...``, see ``compare``.

        Args:
            args: The value node, followed by the list item nodes.

        Returns:
            An ``IN`` expression.
        """

        x = self.operand(args[0])
        if x.isDate or x.isTimeStamp:
            return sa.or_(*[self.compare('=', x, self.operand(a)) for a in args[1:]])
        return self.operand_sql(x).in_([self.build(a) for a in args[1:]])

    def build_not_in(self, args):
        """Build a ``NOT IN`` predicate.

        Args:
            args: The value node, followed by the list item nodes.

        Returns:
            A negated ``IN`` expression.
        """

        return sa.not_(self.build_in(args))

    def build_like(self, args):
        """Build a ``LIKE`` predicate.

        Args:
            args: The value and pattern nodes.

        Returns:
            A ``LIKE`` expression.
        """

        col = self.build(args[0])
        return col.like(self.build(args[1]))

    def build_not_like(self, args):
        """Build a ``NOT LIKE`` predicate.

        Args:
            args: The value and pattern nodes.

        Returns:
            A negated ``LIKE`` expression.
        """

        return sa.not_(self.build_like(args))

    def build_is_null(self, args):
        """Build an ``IS NULL`` predicate.

        Args:
            args: A list with the value node.

        Returns:
            An ``IS NULL`` expression.
        """

        col = self.build(args[0])
        return col.is_(None)

    def build_not_null(self, args):
        """Build an ``IS NOT NULL`` predicate.

        Args:
            args: A list with the value node.

        Returns:
            An ``IS NOT NULL`` expression.
        """

        col = self.build(args[0])
        return col.isnot(None)

    ##

    def func_s_intersects(self, args):
        """Build the ``S_INTERSECTS`` spatial predicate.

        Args:
            args: Two geometry nodes.

        Returns:
            An ``ST_Intersects`` expression.
        """

        return sa.func.ST_Intersects(self.build(args[0]), self.build(args[1]))

    def func_s_contains(self, args):
        """Build the ``S_CONTAINS`` spatial predicate.

        Args:
            args: Two geometry nodes.

        Returns:
            An ``ST_Contains`` expression.
        """

        return sa.func.ST_Contains(self.build(args[0]), self.build(args[1]))

    def func_s_crosses(self, args):
        """Build the ``S_CROSSES`` spatial predicate.

        Args:
            args: Two geometry nodes.

        Returns:
            An ``ST_Crosses`` expression.
        """

        return sa.func.ST_Crosses(self.build(args[0]), self.build(args[1]))

    def func_s_disjoint(self, args):
        """Build the ``S_DISJOINT`` spatial predicate.

        Args:
            args: Two geometry nodes.

        Returns:
            An ``ST_Disjoint`` expression.
        """

        return sa.func.ST_Disjoint(self.build(args[0]), self.build(args[1]))

    def func_s_equals(self, args):
        """Build the ``S_EQUALS`` spatial predicate.

        Args:
            args: Two geometry nodes.

        Returns:
            An ``ST_Equals`` expression.
        """

        return sa.func.ST_Equals(self.build(args[0]), self.build(args[1]))

    def func_s_overlaps(self, args):
        """Build the ``S_OVERLAPS`` spatial predicate.

        Args:
            args: Two geometry nodes.

        Returns:
            An ``ST_Overlaps`` expression.
        """

        return sa.func.ST_Overlaps(self.build(args[0]), self.build(args[1]))

    def func_s_touches(self, args):
        """Build the ``S_TOUCHES`` spatial predicate.

        Args:
            args: Two geometry nodes.

        Returns:
            An ``ST_Touches`` expression.
        """

        return sa.func.ST_Touches(self.build(args[0]), self.build(args[1]))

    def func_s_within(self, args):
        """Build the ``S_WITHIN`` spatial predicate.

        Args:
            args: Two geometry nodes.

        Returns:
            An ``ST_Within`` expression.
        """

        return sa.func.ST_Within(self.build(args[0]), self.build(args[1]))

    ##

    def func_casei(self, args):
        """Build the ``CASEI`` function.

        Args:
            args: A string node.

        Returns:
            The lowercased value.
        """

        return sa.func.lower(self.build(args[0]))

    def func_accenti(self, args):
        """Build the ``ACCENTI`` function.

        Args:
            args: A string node.

        Returns:
            The value without accents, using the postgres ``unaccent`` extension.
        """

        return sa.func.unaccent(self.build(args[0]))

    ##

    def func_bbox(self, args):
        """Build the ``BBOX`` function.

        Args:
            args: Four number literal nodes: min x, min y, max x, max y.

        Returns:
            A rectangle geometry in WGS84.

        Raises:
            BuildError: If an argument is not a literal.
        """

        return self.build_bbox([self.value(a) for a in args])

    def func_timestamp(self, args):
        """Build the ``TIMESTAMP`` function.

        Args:
            args: An ISO timestamp string node, the zone is optional.

        Returns:
            A ``timestamptz`` bound parameter.

        Raises:
            BuildError: If the argument is not a valid timestamp string.
        """

        return self.timestamp_literal(self.timestamp_value(args[0]))

    def func_date(self, args):
        """Build the ``DATE`` function.

        Args:
            args: A date string node, e.g. ``'2021-06-15'``.

        Returns:
            A ``date`` bound parameter.

        Raises:
            BuildError: If the argument is not a valid date string.
        """

        return self.date_literal(self.date_value(args[0]))

    def func_interval(self, args):
        """Reject ``INTERVAL`` outside of temporal predicates.

        Args:
            args: Interval bound nodes.

        Raises:
            BuildError: Always.
        """

        raise BuildError('CQL: INTERVAL is only allowed in temporal predicates')

    ##

    def func_t_equals(self, args):
        """Build the ``T_EQUALS`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if both ranges are equal.
        """

        a, b = self.temporal_pair(args)
        return a == b

    def func_t_after(self, args):
        """Build the ``T_AFTER`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the first range starts after the second ends.
        """

        a, b = self.temporal_pair(args)
        return a.op('>>')(b)

    def func_t_before(self, args):
        """Build the ``T_BEFORE`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the first range ends before the second starts.
        """

        a, b = self.temporal_pair(args)
        return a.op('<<')(b)

    def func_t_meets(self, args):
        """Build the ``T_MEETS`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the first range ends where the second starts.
        """

        a, b = self.temporal_pair(args)
        return sa.func.upper(a) == sa.func.lower(b)

    def func_t_metby(self, args):
        """Build the ``T_METBY`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the first range starts where the second ends.
        """

        a, b = self.temporal_pair(args)
        return sa.func.lower(a) == sa.func.upper(b)

    def func_t_during(self, args):
        """Build the ``T_DURING`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the first range lies strictly inside the second.
        """

        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) > sa.func.lower(b),
            sa.func.upper(a) < sa.func.upper(b),
        )

    def func_t_contains(self, args):
        """Build the ``T_CONTAINS`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the second range lies strictly inside the first.
        """

        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) < sa.func.lower(b),
            sa.func.upper(a) > sa.func.upper(b),
        )

    def func_t_overlaps(self, args):
        """Build the ``T_OVERLAPS`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the first range starts before the second and ends inside it.
        """

        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) < sa.func.lower(b),
            sa.func.upper(a) > sa.func.lower(b),
            sa.func.upper(a) < sa.func.upper(b),
        )

    def func_t_overlappedby(self, args):
        """Build the ``T_OVERLAPPEDBY`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the first range starts inside the second and ends after it.
        """

        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) > sa.func.lower(b),
            sa.func.lower(a) < sa.func.upper(b),
            sa.func.upper(a) > sa.func.upper(b),
        )

    def func_t_starts(self, args):
        """Build the ``T_STARTS`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if both ranges start together and the first ends earlier.
        """

        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) == sa.func.lower(b),
            sa.func.upper(a) < sa.func.upper(b),
        )

    def func_t_startedby(self, args):
        """Build the ``T_STARTEDBY`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if both ranges start together and the first ends later.
        """

        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.lower(a) == sa.func.lower(b),
            sa.func.upper(a) > sa.func.upper(b),
        )

    def func_t_finishes(self, args):
        """Build the ``T_FINISHES`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if both ranges end together and the first starts later.
        """

        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.upper(a) == sa.func.upper(b),
            sa.func.lower(a) > sa.func.lower(b),
        )

    def func_t_finishedby(self, args):
        """Build the ``T_FINISHEDBY`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if both ranges end together and the first starts earlier.
        """

        a, b = self.temporal_pair(args)
        return sa.and_(
            sa.func.upper(a) == sa.func.upper(b),
            sa.func.lower(a) < sa.func.lower(b),
        )

    def func_t_intersects(self, args):
        """Build the ``T_INTERSECTS`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the ranges have a common point.
        """

        a, b = self.temporal_pair(args)
        return a.op('&&')(b)

    def func_t_disjoint(self, args):
        """Build the ``T_DISJOINT`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the ranges have no common point.
        """

        a, b = self.temporal_pair(args)
        return sa.not_(a.op('&&')(b))

    ##

    def func_a_equals(self, args):
        """Build the ``A_EQUALS`` array predicate.

        Args:
            args: Two array nodes.

        Returns:
            A condition that is true if both arrays contain the same elements, ignoring order and duplicates.
        """

        a, b = self.array_pair(args)
        return sa.and_(a.op('@>')(b), a.op('<@')(b))

    def func_a_contains(self, args):
        """Build the ``A_CONTAINS`` array predicate.

        Args:
            args: Two array nodes.

        Returns:
            A condition that is true if the first array contains all elements of the second.
        """

        a, b = self.array_pair(args)
        return a.op('@>')(b)

    def func_a_containedby(self, args):
        """Build the ``A_CONTAINEDBY`` array predicate.

        Args:
            args: Two array nodes.

        Returns:
            A condition that is true if all elements of the first array are in the second.
        """

        a, b = self.array_pair(args)
        return a.op('<@')(b)

    def func_a_overlaps(self, args):
        """Build the ``A_OVERLAPS`` array predicate.

        Args:
            args: Two array nodes.

        Returns:
            A condition that is true if the arrays have a common element.
        """

        a, b = self.array_pair(args)
        return a.op('&&')(b)

    ##

    def compare(self, op, x: _Operand, y: _Operand):
        """Build a comparison.

        A string compared with a temporal operand is read as a date or a timestamp.
        A date and a timestamp are compared by the date of the timestamp.

        Args:
            op: Comparison operator.
            x: Left operand.
            y: Right operand.

        Returns:
            A condition.

        Raises:
            BuildError: If a string is not a valid date or timestamp.
        """

        x_temporal = x.isDate or x.isTimeStamp
        y_temporal = y.isDate or y.isTimeStamp
        if x.node[0] == Node.STRING and not x_temporal and y_temporal:
            x = self.temporal_operand(x)
        if y.node[0] == Node.STRING and not y_temporal and x_temporal:
            y = self.temporal_operand(y)

        if x.isDate and y.isTimeStamp:
            return self.compare(_MIRRORED_OPS[op], y, x)

        fn = _BINARY_OPS[op]

        if x.isTimeStamp and y.isDate:
            if y.value is not None:
                return self.compare_with_day(op, self.operand_sql(x), y.value)
            if x.value is not None:
                return fn(self.date_literal(self.to_date(x.value)), y.expr)
            return fn(self.sql_to_date(x.expr), y.expr)

        return fn(self.operand_sql(x), self.operand_sql(y))

    def compare_with_day(self, op, t, d: datetime.date):
        """Compare a timestamp with a date, using the day boundaries.

        Args:
            op: Comparison operator.
            t: Timestamp expression.
            d: Date.

        Returns:
            A condition on ``t``, e.g. ``t >= d0 AND t < d1`` for ``=``.
        """

        d0, d1 = self.day_bounds(d)
        if op == '=':
            return sa.and_(t >= d0, t < d1)
        if op in C.NOT_EQUAL_OPERATORS:
            return sa.not_(sa.and_(t >= d0, t < d1))
        if op == '>':
            return t >= d1
        if op == '>=':
            return t >= d0
        if op == '<':
            return t < d0
        if op == '<=':
            return t < d1
        raise BuildError(f'CQL: operator {op!r} not implemented')

    def operand(self, e) -> _Operand:
        """Wrap a node as an operand.

        Date and timestamp functions are evaluated, strings are left pending,
        other nodes are built.

        Args:
            e: Node.

        Returns:
            The operand.

        Raises:
            BuildError: If a date or timestamp is invalid.
        """

        if e[0] == Node.STRING:
            return _Operand(node=e)
        if e[0] == Node.FUNCTION and e[1] == 'date':
            return _Operand(node=e, isDate=True, value=self.date_value(e[2]))
        if e[0] == Node.FUNCTION and e[1] == 'timestamp':
            return _Operand(node=e, isTimeStamp=True, value=self.timestamp_value(e[2]))
        x = self.build(e)
        typ = getattr(x, 'type', None)
        return _Operand(
            node=e,
            isDate=isinstance(typ, sa.Date),
            isTimeStamp=isinstance(typ, sa.DateTime),
            expr=x,
        )

    def temporal_operand(self, o: _Operand) -> _Operand:
        """Read a pending string operand as a date or a timestamp.

        Args:
            o: String operand.

        Returns:
            A date operand if the string has no time part, a timestamp operand otherwise.

        Raises:
            BuildError: If the string is not a valid date or timestamp.
        """

        try:
            v = parser.parse_datetime(o.node[1], self.default_time_zone)
        except ParseError as exc:
            raise BuildError(f'CQL: {exc}')
        if isinstance(v, datetime.datetime):
            return _Operand(node=o.node, isTimeStamp=True, value=v)
        return _Operand(node=o.node, isDate=True, value=v)

    def operand_sql(self, o: _Operand):
        """Build the expression of an operand.

        Args:
            o: Operand.

        Returns:
            The built expression, or a bound parameter for a literal.
        """

        if o.expr is not None:
            return o.expr
        if o.isDate:
            return self.date_literal(o.value)
        if o.isTimeStamp:
            return self.timestamp_literal(o.value)
        return self.build(o.node)

    ##

    def date_value(self, e) -> datetime.date:
        """Parse a date string node.

        Args:
            e: String node.

        Returns:
            A ``date``.

        Raises:
            BuildError: If the node is not a string or not a valid date.
        """

        if e[0] != Node.STRING:
            raise BuildError(f'CQL: expected a date string, got {e[0]!r}')
        try:
            return parser.parse_date(e[1])
        except ParseError as exc:
            raise BuildError(f'CQL: {exc}')

    def timestamp_value(self, e) -> datetime.datetime:
        """Parse a timestamp string node, in the default time zone if it has no zone.

        Args:
            e: String node.

        Returns:
            An aware ``datetime``.

        Raises:
            BuildError: If the node is not a string or not a valid timestamp.
        """

        if e[0] != Node.STRING:
            raise BuildError(f'CQL: expected a timestamp string, got {e[0]!r}')
        try:
            return parser.parse_timestamp(e[1], self.default_time_zone)
        except ParseError as exc:
            raise BuildError(f'CQL: {exc}')

    def date_literal(self, d: datetime.date):
        """Wrap a date as a bound parameter.

        Args:
            d: Date.

        Returns:
            A ``date`` bound parameter.
        """

        return sa.literal(d, sa.Date())

    def timestamp_literal(self, t: datetime.datetime):
        """Wrap a datetime as a bound parameter.

        Args:
            t: Aware datetime.

        Returns:
            A ``timestamptz`` bound parameter.
        """

        return sa.literal(t, sa.TIMESTAMP(timezone=True))

    def day_bounds(self, d: datetime.date) -> tuple[datetime.datetime, datetime.datetime]:
        """Compute the start of a day and of the next day.

        Args:
            d: Date.

        Returns:
            A tuple of two aware datetimes in the default time zone.
        """

        n = d + datetime.timedelta(days=1)
        return (
            dtx.new(d.year, d.month, d.day, tz=self.default_time_zone),
            dtx.new(n.year, n.month, n.day, tz=self.default_time_zone),
        )

    def to_date(self, t: datetime.datetime) -> datetime.date:
        """Get the date of a timestamp.

        Args:
            t: Aware datetime.

        Returns:
            The date in the default time zone.
        """

        return t.astimezone(dtx.time_zone(self.default_time_zone)).date()

    def sql_to_date(self, x):
        """Build the date of a timestamp expression.

        Args:
            x: ``timestamptz`` expression.

        Returns:
            A ``date`` expression, in the default time zone.
        """

        return sa.cast(sa.func.timezone(self.time_zone_name(), x), sa.Date())

    def sql_day_start(self, x, days=0):
        """Build the start of the day of a date expression.

        Args:
            x: ``date`` expression.
            days: Number of days to add.

        Returns:
            A ``timestamptz`` expression, in the default time zone.
        """

        if days:
            x = x + days
        return sa.func.timezone(self.time_zone_name(), sa.cast(x, sa.TIMESTAMP()), type_=sa.TIMESTAMP(timezone=True))

    def time_zone_name(self):
        """Get the name of the default time zone.

        Returns:
            A time zone name for postgres, e.g. ``Europe/Berlin``.
        """

        return dtx.time_zone(self.default_time_zone).key

    ##

    def temporal_pair(self, args):
        """Coerce both operands of a temporal predicate to ranges.

        Args:
            args: Two temporal nodes.

        Returns:
            A tuple of two ``tstzrange`` expressions.
        """

        return self.temporal_range(args[0]), self.temporal_range(args[1])

    def temporal_range(self, e):
        """Coerce a temporal expression to a ``tstzrange``.

        A timestamp becomes a degenerate range, a date the half-open range of its day.
        A null bound is unbounded in postgres, therefore null inputs must yield a null range,
        otherwise a null column would match everything.

        Args:
            e: An ``INTERVAL`` function node, or a date or timestamp node.

        Returns:
            A ``tstzrange`` expression, null if a bound is null.
        """

        if e[0] == Node.FUNCTION and e[1] == 'interval':
            lo, _, _ = self.temporal_span(e[2], '-infinity')
            _, hi, hi_open = self.temporal_span(e[3], 'infinity')
        else:
            lo, hi, hi_open = self.temporal_span(e, '')

        return sa.case(
            (sa.or_(lo.is_(None), hi.is_(None)), sa.null()),
            else_=sa.func.tstzrange(lo, hi, '[)' if hi_open else '[]'),
        )

    def temporal_span(self, e, unbounded):
        """Build the start and end of an instant or an interval bound.

        Args:
            e: Date, timestamp or string node. In an interval, the string ``'..'`` means an open bound.
            unbounded: Value for an open bound, ``-infinity`` or ``infinity``, empty outside of intervals.

        Returns:
            A tuple of the start and end expressions and a flag that tells whether the end is open.

        Raises:
            BuildError: If the node is not a date or a timestamp.
        """

        if unbounded and e[0] == Node.STRING and e[1] == '..':
            x = sa.cast(sa.literal(unbounded), sa.TIMESTAMP(timezone=True))
            return x, x, False

        o = self.operand(e)
        if o.node[0] == Node.STRING and not o.isDate and not o.isTimeStamp:
            o = self.temporal_operand(o)

        if o.isDate:
            if o.value is not None:
                d0, d1 = self.day_bounds(o.value)
                return self.timestamp_literal(d0), self.timestamp_literal(d1), True
            return self.sql_day_start(o.expr), self.sql_day_start(o.expr, 1), True

        if o.isTimeStamp:
            x = self.operand_sql(o)
            typ = getattr(x, 'type', None)
            if isinstance(typ, sa.DateTime) and not typ.timezone:
                x = sa.func.timezone(self.time_zone_name(), x, type_=sa.TIMESTAMP(timezone=True))
            return x, x, False

        raise BuildError(f'CQL: expected a date or a timestamp, got {e[0]!r}')

    ##

    def literal(self, val):
        """Wrap a python value as a bound parameter.

        Args:
            val: Value.

        Returns:
            A literal expression.
        """

        return sa.literal(val)

    ##

    def array_pair(self, args):
        """Coerce both operands of an array predicate to arrays.

        Args:
            args: Two array nodes.

        Returns:
            A tuple of two array expressions.
        """

        return self.array_operand(args[0]), self.array_operand(args[1])

    def array_operand(self, e):
        """Build an array expression, an array literal becoming a typed parameter.

        Args:
            e: Array literal or another node.

        Returns:
            An array expression.
        """

        if e[0] != Node.ARRAY:
            return self.build(e)
        vals = self.value(e)
        return sa.literal(vals, sa.ARRAY(self.array_element_type(vals)))

    def array_element_type(self, vals):
        """Infer the element type of an array literal from its first element.

        Args:
            vals: Array values.

        Returns:
            An SQLAlchemy type, ``Text`` for an empty array or non-numeric values.
        """

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
