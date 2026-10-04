"""Build database expressions from CQL2 parse trees."""

from typing import Any, cast

import operator

import gws
import gws.lib.crs
import gws.lib.datetimex as dtx
import gws.lib.sa as sa

from .parser import Node, C


class BuildError(Exception):
    """A parse tree node or function is not supported by the builder, or its arguments are invalid."""

    pass


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


class SqlBuilder(Builder):
    """Builds SQLAlchemy expressions for a postgis table.

    Names refer to table columns, literals become bound parameters, geometries are in WGS84.
    All standard CQL2 functions are implemented, user functions are not.
    """

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
        """Create a builder.

        Args:
            table: Table whose columns are referenced by names.
        """
        self.table = table

    def build_operator(self, op, args):
        fn = self._binary_ops.get(op)
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

    def build_date(self, args):
        """Build a date literal.

        Args:
            args: A list with the date value.

        Returns:
            A ``DATE`` expression.
        """
        return sa.cast(args[0], sa.DATE())

    def build_timestamp(self, args):
        """Build a timestamp literal.

        Args:
            args: A list with the datetime value.

        Returns:
            A ``timestamptz`` expression.
        """
        return sa.cast(args[0], sa.TIMESTAMP(timezone=True))

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

        Args:
            args: The value, lower bound and upper bound nodes.

        Returns:
            A ``BETWEEN`` expression.
        """
        col = self.build(args[0])
        a = self.build(args[1])
        b = self.build(args[2])
        return col.between(a, b)

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

        Args:
            args: The value node, followed by the list item nodes.

        Returns:
            An ``IN`` expression.
        """
        col = self.build(args[0])
        ls = [self.build(a) for a in args[1:]]
        return col.in_(ls)

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
            args: An ISO date-time string literal node.

        Returns:
            A ``timestamptz`` expression.

        Raises:
            BuildError: If the argument is not a literal.
        """
        dt = dtx.from_iso_string(self.value(args[0]), 'UTC')
        return sa.cast(dt, sa.TIMESTAMP(timezone=True))

    def func_date(self, args):
        """Build the ``DATE`` function.

        Args:
            args: An ISO date string literal node.

        Returns:
            A ``DATE`` expression.

        Raises:
            BuildError: If the argument is not a literal.
        """
        dt = dtx.from_iso_string(self.value(args[0]), 'UTC')
        return sa.cast(dt, sa.DATE())

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
        return sa.func.lower(a) > sa.func.upper(b)

    def func_t_before(self, args):
        """Build the ``T_BEFORE`` temporal predicate.

        Args:
            args: Two temporal nodes, instants or intervals.

        Returns:
            A condition that is true if the first range ends before the second starts.
        """
        a, b = self.temporal_pair(args)
        return sa.func.upper(a) < sa.func.lower(b)

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

    def temporal_pair(self, args):
        """Coerce both operands of a temporal predicate to ranges.

        Args:
            args: Two temporal nodes.

        Returns:
            A tuple of two ``tstzrange`` expressions.
        """

        return self.temporal_range(args[0]), self.temporal_range(args[1])

    def temporal_range(self, e):
        """Coerce a temporal expression to a ``tstzrange``, an instant becomes a degenerate range.

        A null bound is unbounded in postgres, therefore null inputs must yield a null range,
        otherwise a null column would match everything.

        Args:
            e: An ``INTERVAL`` function node, or a node that builds a timestamp.

        Returns:
            A ``tstzrange`` expression with inclusive bounds, null if a bound is null.
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
        """Build an interval bound.

        Args:
            e: Bound node. The string ``'..'`` means an open bound.
            unbounded: Value for an open bound, ``-infinity`` or ``infinity``.

        Returns:
            A ``timestamptz`` expression.
        """

        if e[0] == Node.STRING and e[1] == '..':
            return sa.cast(sa.literal(unbounded), sa.TIMESTAMP(timezone=True))
        return self.timestamp_value(e)

    def timestamp_value(self, e):
        """Coerce an expression to a ``timestamptz``, naive values are assumed to be UTC.

        Args:
            e: Node.

        Returns:
            A ``timestamptz`` expression.
        """

        x = self.build(e)
        typ = getattr(x, 'type', None)
        if isinstance(typ, sa.TIMESTAMP) and typ.timezone:
            return x
        return sa.cast(sa.func.timezone('UTC', sa.cast(x, sa.TIMESTAMP())), sa.TIMESTAMP(timezone=True))

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
