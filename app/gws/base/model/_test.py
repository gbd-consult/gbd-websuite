"""Tests for the model utilities."""

import gws
import gws.base.model.util as util
import gws.lib.crs
import gws.lib.shape


def test_describe_from_record_columns():
    fd = gws.FeatureRecord(attributes={'a': 'str_1', 'b': 1, 'c': 1.5}, shape=None)
    desc = util.describe_from_record(fd)

    assert [c.name for c in desc.columns] == ['a', 'b', 'c']
    assert [c.type for c in desc.columns] == [gws.AttributeType.str, gws.AttributeType.int, gws.AttributeType.float]
    assert not desc.geometryName


def test_describe_from_record_geometry_once():
    shape = gws.lib.shape.from_xy(1, 2, gws.lib.crs.WEBMERCATOR)
    fd = gws.FeatureRecord(attributes={'a': 'str_1', 'b': 1}, shape=shape)
    desc = util.describe_from_record(fd)

    assert [c.name for c in desc.columns] == ['a', 'b', 'geometry']
    assert desc.geometryName == 'geometry'
    assert desc.geometryType == gws.GeometryType.point
    assert desc.geometrySrid == 3857


def test_describe_from_record_geometry_without_attributes():
    shape = gws.lib.shape.from_xy(1, 2, gws.lib.crs.WEBMERCATOR)
    fd = gws.FeatureRecord(attributes={}, shape=shape)
    desc = util.describe_from_record(fd)

    assert [c.name for c in desc.columns] == ['geometry']
    assert set(desc.columnMap) == {'geometry'}
