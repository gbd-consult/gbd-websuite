"""Tests for the cql parser module."""

import datetime

import pytest

import gws
import gws.test.util as u
import gws.lib.cql as cql


def test_simple_equality():
    result = cql.parse("name = 'John'")
    assert result == ['=', [cql.Node.NAME, 'name'], [cql.Node.STRING, 'John']]


def test_numeric_comparison():
    result = cql.parse('age > 25')
    assert result == ['>', [cql.Node.NAME, 'age'], [cql.Node.INT, 25]]


def test_and_expression():
    result = cql.parse("age > 18 AND status = 'active'")
    assert result == [
        cql.Node.AND,
        ['>', [cql.Node.NAME, 'age'], [cql.Node.INT, 18]],
        ['=', [cql.Node.NAME, 'status'], [cql.Node.STRING, 'active']],
    ]


def test_or_expression():
    result = cql.parse("city = 'NYC' OR city = 'LA'")
    assert result == [
        cql.Node.OR,
        ['=', [cql.Node.NAME, 'city'], [cql.Node.STRING, 'NYC']],
        ['=', [cql.Node.NAME, 'city'], [cql.Node.STRING, 'LA']],
    ]


def test_not_expression():
    result = cql.parse('NOT active = true')
    assert result == [cql.Node.NOT, ['=', [cql.Node.NAME, 'active'], [cql.Node.BOOL, True]]]


def test_parentheses():
    result = cql.parse('(age > 18 AND age < 65) OR retired = true')
    assert result[0] == cql.Node.OR
    assert result[1] == [
        cql.Node.AND,
        ['>', [cql.Node.NAME, 'age'], [cql.Node.INT, 18]],
        ['<', [cql.Node.NAME, 'age'], [cql.Node.INT, 65]],
    ]


def test_like_predicate():
    result = cql.parse("name LIKE 'John%'")
    assert result == [cql.Node.LIKE, [cql.Node.NAME, 'name'], [cql.Node.STRING, 'John%']]


def test_between_predicate():
    result = cql.parse('age BETWEEN 18 AND 65')
    assert result == [cql.Node.BETWEEN, [cql.Node.NAME, 'age'], [cql.Node.INT, 18], [cql.Node.INT, 65]]


def test_in_predicate_with_parens():
    result = cql.parse("status IN ('active', 'pending')")
    assert result == [cql.Node.IN, [cql.Node.NAME, 'status'], [cql.Node.STRING, 'active'], [cql.Node.STRING, 'pending']]


def test_is_null_predicate():
    result = cql.parse('middle_name IS NULL')
    assert result == [cql.Node.IS_NULL, [cql.Node.NAME, 'middle_name']]


def test_arithmetic_addition():
    result = cql.parse('price + tax = 100')
    assert result == ['=', ['+', [cql.Node.NAME, 'price'], [cql.Node.NAME, 'tax']], [cql.Node.INT, 100]]


def test_float_number():
    result = cql.parse('price = 19.99')
    assert result == ['=', [cql.Node.NAME, 'price'], [cql.Node.FLOAT, 19.99]]


def test_date_literal():
    result = cql.parse('birthdate = 2000-01-15')
    assert result == ['=', [cql.Node.NAME, 'birthdate'], [cql.Node.DATE, datetime.date(2000, 1, 15)]]


def test_invalid_syntax():
    with pytest.raises(cql.ParseError):
        cql.parse('name =')


def test_unexpected_token():
    with pytest.raises(cql.ParseError):
        cql.parse("name = 'test' extra")


def test_integer_positive():
    result = cql.parse('count = 42')
    assert result == ['=', [cql.Node.NAME, 'count'], [cql.Node.INT, 42]]


def test_integer_negative():
    result = cql.parse('temperature = -15')
    assert result == ['=', [cql.Node.NAME, 'temperature'], [cql.Node.INT, -15]]


def test_integer_zero():
    result = cql.parse('balance = 0')
    assert result == ['=', [cql.Node.NAME, 'balance'], [cql.Node.INT, 0]]


def test_float_positive():
    result = cql.parse('price = 123.45')
    assert result == ['=', [cql.Node.NAME, 'price'], [cql.Node.FLOAT, 123.45]]


def test_float_negative():
    result = cql.parse('change = -9.99')
    assert result == ['=', [cql.Node.NAME, 'change'], [cql.Node.FLOAT, -9.99]]


def test_float_scientific_notation():
    result = cql.parse('distance = 1.5e10')
    assert result == ['=', [cql.Node.NAME, 'distance'], [cql.Node.FLOAT, 1.5e10]]


def test_float_scientific_negative_exponent():
    result = cql.parse('precision = 3.14e-5')
    assert result == ['=', [cql.Node.NAME, 'precision'], [cql.Node.FLOAT, 3.14e-5]]


def test_string_simple():
    result = cql.parse("name = 'Alice'")
    assert result == ['=', [cql.Node.NAME, 'name'], [cql.Node.STRING, 'Alice']]


def test_string_with_spaces():
    result = cql.parse("full_name = 'John Doe'")
    assert result == ['=', [cql.Node.NAME, 'full_name'], [cql.Node.STRING, 'John Doe']]


def test_string_empty():
    result = cql.parse("note = ''")
    assert result == ['=', [cql.Node.NAME, 'note'], [cql.Node.STRING, '']]


def test_string_with_escaped_quote():
    result = cql.parse("text = 'It\\'s'")
    assert result == ['=', [cql.Node.NAME, 'text'], [cql.Node.STRING, "It's"]]


def test_string_with_doubled_quote():
    result = cql.parse("text = 'It''s'")
    assert result == ['=', [cql.Node.NAME, 'text'], [cql.Node.STRING, "It's"]]


def test_string_with_doubled_quote_only():
    result = cql.parse("text = ''''")
    assert result == ['=', [cql.Node.NAME, 'text'], [cql.Node.STRING, "'"]]


def test_quoted_name_with_doubled_quote():
    result = cql.parse('"a""b" = 1')
    assert result == ['=', [cql.Node.NAME, 'a"b'], [cql.Node.INT, 1]]


def test_string_with_backslash():
    result = cql.parse("text = 'Line1\\nLine2'")
    assert result == ['=', [cql.Node.NAME, 'text'], [cql.Node.STRING, 'Line1\\nLine2']]


def test_string_with_non_ascii():
    result = cql.parse("text = 'Straße'")
    assert result == ['=', [cql.Node.NAME, 'text'], [cql.Node.STRING, 'Straße']]


def test_quoted_name_with_non_ascii():
    result = cql.parse('"Größe" = 1')
    assert result == ['=', [cql.Node.NAME, 'Größe'], [cql.Node.INT, 1]]


def test_date_literal_basic():
    result = cql.parse('created_at = 2025-12-09')
    assert result == ['=', [cql.Node.NAME, 'created_at'], [cql.Node.DATE, datetime.date(2025, 12, 9)]]


def test_date_literal_year_boundary():
    result = cql.parse('new_year = 2024-01-01')
    assert result == ['=', [cql.Node.NAME, 'new_year'], [cql.Node.DATE, datetime.date(2024, 1, 1)]]


def test_date_literal_leap_year():
    result = cql.parse('leap_day = 2024-02-29')
    assert result == ['=', [cql.Node.NAME, 'leap_day'], [cql.Node.DATE, datetime.date(2024, 2, 29)]]


def test_boolean_true():
    result = cql.parse('active = TRUE')
    assert result == ['=', [cql.Node.NAME, 'active'], [cql.Node.BOOL, True]]


def test_boolean_false():
    result = cql.parse('disabled = FALSE')
    assert result == ['=', [cql.Node.NAME, 'disabled'], [cql.Node.BOOL, False]]


def test_boolean_true_lowercase():
    result = cql.parse('enabled = true')
    assert result == ['=', [cql.Node.NAME, 'enabled'], [cql.Node.BOOL, True]]


def test_mixed_literals_in_list():
    result = cql.parse("value IN (1, 2.5, 'text')")
    assert result == [cql.Node.IN, [cql.Node.NAME, 'value'], [cql.Node.INT, 1], [cql.Node.FLOAT, 2.5], [cql.Node.STRING, 'text']]


def test_date_in_comparison():
    result = cql.parse('event_date > 2025-01-01')
    assert result == ['>', [cql.Node.NAME, 'event_date'], [cql.Node.DATE, datetime.date(2025, 1, 1)]]


def test_multiple_numeric_types():
    result = cql.parse('total = 100 + 25.5')
    assert result == ['=', [cql.Node.NAME, 'total'], ['+', [cql.Node.INT, 100], [cql.Node.FLOAT, 25.5]]]


def test_not_like_predicate():
    result = cql.parse("name NOT LIKE '%test%'")
    assert result == [cql.Node.NOT_LIKE, [cql.Node.NAME, 'name'], [cql.Node.STRING, '%test%']]


def test_not_between_predicate():
    result = cql.parse('age NOT BETWEEN 0 AND 10')
    assert result == [cql.Node.NOT_BETWEEN, [cql.Node.NAME, 'age'], [cql.Node.INT, 0], [cql.Node.INT, 10]]


def test_not_in_predicate():
    result = cql.parse("status NOT IN ('pending', 'draft')")
    assert result == [cql.Node.NOT_IN, [cql.Node.NAME, 'status'], [cql.Node.STRING, 'pending'], [cql.Node.STRING, 'draft']]


def test_is_not_null_predicate():
    result = cql.parse('email IS NOT NULL')
    assert result == [cql.Node.NOT_NULL, [cql.Node.NAME, 'email']]


def test_in_predicate_with_brackets():
    result = cql.parse('id IN [1, 2, 3]')
    assert result == [cql.Node.IN, [cql.Node.NAME, 'id'], [cql.Node.INT, 1], [cql.Node.INT, 2], [cql.Node.INT, 3]]


def test_array_literal():
    result = cql.parse('tags = [1, 2, 3]')
    assert result == ['=', [cql.Node.NAME, 'tags'], [cql.Node.ARRAY, [cql.Node.INT, 1], [cql.Node.INT, 2], [cql.Node.INT, 3]]]


def test_empty_array():
    result = cql.parse('items = []')
    assert result == ['=', [cql.Node.NAME, 'items'], [cql.Node.ARRAY]]


def test_arithmetic_subtraction():
    result = cql.parse('total = 100 - 25')
    assert result == ['=', [cql.Node.NAME, 'total'], ['-', [cql.Node.INT, 100], [cql.Node.INT, 25]]]


def test_arithmetic_multiplication():
    result = cql.parse('area = width * height')
    assert result == ['=', [cql.Node.NAME, 'area'], ['*', [cql.Node.NAME, 'width'], [cql.Node.NAME, 'height']]]


def test_arithmetic_division():
    result = cql.parse('average = sum / count')
    assert result == ['=', [cql.Node.NAME, 'average'], ['/', [cql.Node.NAME, 'sum'], [cql.Node.NAME, 'count']]]


def test_arithmetic_modulo():
    result = cql.parse('remainder = value % 10')
    assert result == ['=', [cql.Node.NAME, 'remainder'], ['%', [cql.Node.NAME, 'value'], [cql.Node.INT, 10]]]


def test_arithmetic_power():
    result = cql.parse('square = x ^ 2')
    assert result == ['=', [cql.Node.NAME, 'square'], ['^', [cql.Node.NAME, 'x'], [cql.Node.INT, 2]]]


def test_unary_plus():
    result = cql.parse('value = +5')
    assert result == ['=', [cql.Node.NAME, 'value'], [cql.Node.INT, 5]]


def test_nested_arithmetic():
    result = cql.parse('result = (a + b) * c')
    assert result == ['=', [cql.Node.NAME, 'result'], ['*', ['+', [cql.Node.NAME, 'a'], [cql.Node.NAME, 'b']], [cql.Node.NAME, 'c']]]


def test_parenthesized_arithmetic_operand():
    result = cql.parse('(a + b) * c = 1')
    assert result == ['=', ['*', ['+', [cql.Node.NAME, 'a'], [cql.Node.NAME, 'b']], [cql.Node.NAME, 'c']], [cql.Node.INT, 1]]


def test_parenthesized_predicate_operand():
    result = cql.parse("(a) LIKE 'x%'")
    assert result == [cql.Node.LIKE, [cql.Node.NAME, 'a'], [cql.Node.STRING, 'x%']]


def test_parenthesized_comparison_operand():
    result = cql.parse('(a = 1) = true')
    assert result == ['=', ['=', [cql.Node.NAME, 'a'], [cql.Node.INT, 1]], [cql.Node.BOOL, True]]


def test_parenthesized_boolean_operand():
    result = cql.parse('(a = 1 OR b = 2) = true')
    assert result == [
        '=',
        [cql.Node.OR, ['=', [cql.Node.NAME, 'a'], [cql.Node.INT, 1]], ['=', [cql.Node.NAME, 'b'], [cql.Node.INT, 2]]],
        [cql.Node.BOOL, True],
    ]


def test_parenthesized_group_stays_boolean():
    result = cql.parse('(a = 1) AND (b = 2)')
    assert result == [cql.Node.AND, ['=', [cql.Node.NAME, 'a'], [cql.Node.INT, 1]], ['=', [cql.Node.NAME, 'b'], [cql.Node.INT, 2]]]


def test_function_call_no_args():
    result = cql.parse('now() > timestamp')
    assert result == ['>', [cql.Node.USER_FUNCTION, 'now'], [cql.Node.NAME, 'timestamp']]


def test_function_call_with_args():
    result = cql.parse('S_INTERSECTS(geom, point)')
    assert result == [cql.Node.FUNCTION, 's_intersects', [cql.Node.NAME, 'geom'], [cql.Node.NAME, 'point']]


def test_function_call_multiple_args():
    result = cql.parse("func(1, 'test', true)")
    assert result == [cql.Node.USER_FUNCTION, 'func', [cql.Node.INT, 1], [cql.Node.STRING, 'test'], [cql.Node.BOOL, True]]


def test_quoted_identifier():
    result = cql.parse('"column name" = 5')
    assert result == ['=', [cql.Node.NAME, 'column name'], [cql.Node.INT, 5]]


def test_dotted_property_name():
    result = cql.parse("user.address.city = 'NYC'")
    assert result == ['=', [cql.Node.NAME, 'user', 'address', 'city'], [cql.Node.STRING, 'NYC']]


def test_quoted_and_unquoted_in_name():
    result = cql.parse('table."column name".field = 1')
    assert result == ['=', [cql.Node.NAME, 'table', 'column name', 'field'], [cql.Node.INT, 1]]


def test_timestamp_with_z():
    result = cql.parse('created = 2025-12-09T10:30:00Z')
    expected_dt = datetime.datetime.fromisoformat('2025-12-09T10:30:00+00:00')
    assert result == ['=', [cql.Node.NAME, 'created'], [cql.Node.TIMESTAMP, expected_dt]]


def test_timestamp_without_z():
    result = cql.parse('modified = 2025-12-09T14:45:30')
    expected_dt = datetime.datetime.fromisoformat('2025-12-09T14:45:30')
    assert result == ['=', [cql.Node.NAME, 'modified'], [cql.Node.TIMESTAMP, expected_dt]]


def test_timestamp_with_milliseconds():
    result = cql.parse('event_time = 2025-12-09T10:30:00.123Z')
    expected_dt = datetime.datetime.fromisoformat('2025-12-09T10:30:00.123+00:00')
    assert result == ['=', [cql.Node.NAME, 'event_time'], [cql.Node.TIMESTAMP, expected_dt]]


def test_invalid_timestamp():
    with pytest.raises(cql.ParseError):
        cql.parse('time = 2025-13-45T99:99:99Z')


def test_invalid_date():
    with pytest.raises(cql.ParseError):
        cql.parse('date = 2025-13-45')


def test_unexpected_character():
    with pytest.raises(cql.ParseError):
        cql.parse('name = @invalid')


def test_missing_closing_paren():
    with pytest.raises(cql.ParseError):
        cql.parse('(age > 18')


def test_missing_closing_bracket():
    with pytest.raises(cql.ParseError):
        cql.parse('items = [1, 2, 3')


def test_in_without_parens_or_brackets():
    with pytest.raises(cql.ParseError):
        cql.parse("status IN 'active'")


def test_empty_expression():
    with pytest.raises(cql.ParseError):
        cql.parse('')


def test_comparison_operators():
    assert cql.parse('a = b')[0] == '='
    assert cql.parse('a <> b')[0] == '<>'
    assert cql.parse('a != b')[0] == '<>'  # treat != as <>
    assert cql.parse('a < b')[0] == '<'
    assert cql.parse('a <= b')[0] == '<='
    assert cql.parse('a > b')[0] == '>'
    assert cql.parse('a >= b')[0] == '>='


def test_complex_boolean_expression():
    result = cql.parse('(a = 1 OR b = 2) AND (c = 3 OR d = 4)')
    assert result[0] == cql.Node.AND
    assert len(result) == 3


def test_multiple_not():
    result = cql.parse('NOT NOT active = true')
    assert result == [cql.Node.NOT, [cql.Node.NOT, ['=', [cql.Node.NAME, 'active'], [cql.Node.BOOL, True]]]]


def test_geometry_point():
    result = cql.parse('location = POINT(1.0 2.0)')
    assert result == ['=', [cql.Node.NAME, 'location'], [cql.Node.WKT, 'POINT(1.0 2.0)']]


def test_geometry_point_with_z():
    result = cql.parse('location = POINT Z(1.0 2.0 3.0)')
    assert result == ['=', [cql.Node.NAME, 'location'], [cql.Node.WKT, 'POINT Z(1.0 2.0 3.0)']]


def test_geometry_linestring():
    result = cql.parse('path = LINESTRING(0 0, 1 1, 2 2)')
    assert result == ['=', [cql.Node.NAME, 'path'], [cql.Node.WKT, 'LINESTRING(0 0, 1 1, 2 2)']]


def test_geometry_polygon():
    result = cql.parse('area = POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')
    assert result == ['=', [cql.Node.NAME, 'area'], [cql.Node.WKT, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))']]


def test_geometry_polygon_with_hole():
    result = cql.parse('area = POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))')
    assert result == ['=', [cql.Node.NAME, 'area'], [cql.Node.WKT, 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))']]


def test_geometry_multipoint():
    result = cql.parse('locations = MULTIPOINT((1 1), (2 2))')
    assert result == ['=', [cql.Node.NAME, 'locations'], [cql.Node.WKT, 'MULTIPOINT((1 1), (2 2))']]


def test_geometry_multilinestring():
    result = cql.parse('paths = MULTILINESTRING((0 0, 1 1), (2 2, 3 3))')
    assert result == ['=', [cql.Node.NAME, 'paths'], [cql.Node.WKT, 'MULTILINESTRING((0 0, 1 1), (2 2, 3 3))']]


def test_geometry_multipolygon():
    result = cql.parse('areas = MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))')
    assert result == ['=', [cql.Node.NAME, 'areas'], [cql.Node.WKT, 'MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))']]


def test_geometry_collection():
    result = cql.parse('geoms = GEOMETRYCOLLECTION(POINT(1 1), LINESTRING(0 0, 1 1))')
    assert result == ['=', [cql.Node.NAME, 'geoms'], [cql.Node.WKT, 'GEOMETRYCOLLECTION(POINT(1 1), LINESTRING(0 0, 1 1))']]


def test_geometry_negative_coordinates():
    result = cql.parse('location = POINT(-122.5 37.8)')
    assert result == ['=', [cql.Node.NAME, 'location'], [cql.Node.WKT, 'POINT(-122.5 37.8)']]


def test_geometry_in_spatial_function():
    result = cql.parse('S_INTERSECTS(geom, POINT(1 2))')
    assert result == [cql.Node.FUNCTION, 's_intersects', [cql.Node.NAME, 'geom'], [cql.Node.WKT, 'POINT(1 2)']]


def test_geometry_variable_name_point():
    result = cql.parse('point = POINT(1 2)')
    assert result == ['=', [cql.Node.NAME, 'point'], [cql.Node.WKT, 'POINT(1 2)']]


def test_geometry_variable_name_polygon():
    result = cql.parse('polygon = 5')
    assert result == ['=', [cql.Node.NAME, 'polygon'], [cql.Node.INT, 5]]


def test_geometry_variable_name_linestring():
    result = cql.parse("linestring = 'test'")
    assert result == ['=', [cql.Node.NAME, 'linestring'], [cql.Node.STRING, 'test']]


def test_spatial_predicate_s_intersects():
    result = cql.parse('S_INTERSECTS(geom1, POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))) = true')
    assert result[0] == '='
    assert result[1][0] == cql.Node.FUNCTION
    assert result[1][1] == 's_intersects'
    assert result[2] == [cql.Node.BOOL, True]


def test_spatial_predicate_s_contains():
    result = cql.parse('S_CONTAINS(area, POINT(5 5)) = true')
    assert result == [
        '=',
        [cql.Node.FUNCTION, 's_contains', [cql.Node.NAME, 'area'], [cql.Node.WKT, 'POINT(5 5)']],
        [cql.Node.BOOL, True],
    ]


def test_spatial_predicate_s_within():
    result = cql.parse('S_WITHIN(location, boundary) = true')
    assert result == [
        '=',
        [cql.Node.FUNCTION, 's_within', [cql.Node.NAME, 'location'], [cql.Node.NAME, 'boundary']],
        [cql.Node.BOOL, True],
    ]


def test_geometry_with_decimal_coordinates():
    result = cql.parse('coords = POINT(1.234 5.678)')
    assert result == ['=', [cql.Node.NAME, 'coords'], [cql.Node.WKT, 'POINT(1.234 5.678)']]


def test_geometry_in_comparison():
    result = cql.parse('shape = POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))')
    assert result[0] == '='
    assert result[1] == [cql.Node.NAME, 'shape']
    assert result[2][0] == cql.Node.WKT
    assert 'POLYGON' in result[2][1]


def test_geometry_in_boolean_expression():
    result = cql.parse('S_INTERSECTS(geom, POINT(0 0)) AND active = true')
    assert result[0] == cql.Node.AND
    assert result[1][0] == cql.Node.FUNCTION
    assert result[2] == ['=', [cql.Node.NAME, 'active'], [cql.Node.BOOL, True]]
