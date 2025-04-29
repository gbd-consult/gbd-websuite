"""Tests for the MapServer module."""

import io
import os
import pytest
from unittest.mock import patch, MagicMock

import mapscript

import gws
import gws.gis.ms
import gws.lib.image
from gws.test import util


@pytest.fixture
def mock_mapscript_map():
    """Fixture to create a mock mapscript.mapObj."""
    mock_map = MagicMock()
    mock_map.numlayers = 0
    mock_map.setOutputFormat.return_value = None
    mock_map.insertLayer.return_value = None
    mock_map.setExtent.return_value = None
    mock_map.setSize.return_value = None
    mock_map.setProjection.return_value = None

    mock_output_format = MagicMock()
    mock_map.outputformat = mock_output_format

    mock_map.clone.return_value = mock_map
    mock_map.convertToString.return_value = "MAP\nEND"

    return mock_map


@pytest.fixture
def mock_mapscript_layer():
    """Fixture to create a mock mapscript.layerObj."""
    mock_layer = MagicMock()
    mock_layer.setProjection.return_value = None
    mock_layer.addProcessing.return_value = None
    mock_layer.setConnectionType.return_value = None
    mock_layer.updateFromString.return_value = None

    return mock_layer


def test_version():
    """Test the version function returns a string."""
    with patch('mapscript.msGetVersion', return_value="MapServer version 7.6.0"):
        version = gws.gis.ms.version()
        assert isinstance(version, str)
        assert "MapServer" in version


@patch('mapscript.mapObj')
def test_new_map(mock_map_obj):
    """Test creating a new Map instance."""
    mock_map_instance = MagicMock()
    mock_map_obj.return_value = mock_map_instance

    map_instance = gws.gis.ms.new_map()

    assert isinstance(map_instance, gws.gis.ms.Map)
    mock_map_obj.assert_called_once()
    mock_map_instance.setOutputFormat.assert_called_once()
    assert mock_map_instance.outputformat.transparent == mapscript.MS_TRUE


@patch('mapscript.mapObj')
@patch('mapscript.outputFormatObj')
def test_map_init(mock_output_format, mock_map_obj):
    """Test Map initialization."""
    mock_map_instance = MagicMock()
    mock_map_obj.return_value = mock_map_instance
    mock_format = MagicMock()
    mock_output_format.return_value = mock_format

    map_instance = gws.gis.ms.Map()

    assert map_instance.mapObj == mock_map_instance
    mock_map_instance.setOutputFormat.assert_called_once_with(mock_format)
    assert mock_map_instance.outputformat.transparent == mapscript.MS_TRUE


def test_map_copy(mock_mapscript_map):
    """Test copying a Map instance."""
    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()
        copied_map = map_instance.copy()

        assert isinstance(copied_map, gws.gis.ms.Map)
        assert copied_map.mapObj == mock_mapscript_map
        mock_mapscript_map.clone.assert_called_once()


@patch('mapscript.layerObj')
def test_add_raster_layer(mock_layer_obj, mock_mapscript_map):
    """Test adding a raster layer to the map."""
    mock_layer = MagicMock()
    mock_layer_obj.return_value = mock_layer

    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()

        opts = gws.gis.ms.RasterLayerOptions(
            path="/path/to/raster.tif",
            tileIndex="/path/to/tileindex.shp",
            bounds=gws.Bounds(extent=[0, 0, 100, 100]),
            crs=gws.Data(srid=4326),
            processing=["RESAMPLE=BILINEAR", "SCALE=AUTO"]
        )

        layer = map_instance.add_raster_layer(opts)

        assert layer == mock_layer
        mock_layer_obj.assert_called_once_with(mock_mapscript_map)
        assert mock_layer.name == "_gws_0"
        assert mock_layer.type == mapscript.MS_LAYER_RASTER
        assert mock_layer.status == mapscript.MS_ON
        assert mock_layer.data == "/path/to/raster.tif"
        assert mock_layer.tileindex == "/path/to/tileindex.shp"
        assert mock_layer.addProcessing.call_count == 2
        mock_layer.setProjection.assert_called_once_with("init=epsg:4326")
        mock_mapscript_map.insertLayer.assert_called_once_with(mock_layer)


@patch('mapscript.layerObj')
def test_add_vector_layer_point(mock_layer_obj, mock_mapscript_map):
    """Test adding a point vector layer to the map."""
    mock_layer = MagicMock()
    mock_layer_obj.return_value = mock_layer

    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()

        opts = gws.gis.ms.VectorLayerOptions(
            geometryType=gws.GeometryType.point,
            connectionType=None,
            connectionString=None,
            dataString="the_geom from points",
            crs=gws.Data(srid=4326),
            style=None,
            config=None
        )

        layer = map_instance.add_vector_layer(opts)

        assert layer == mock_layer
        mock_layer_obj.assert_called_once_with(mock_mapscript_map)
        assert mock_layer.name == "_gws_0"
        assert mock_layer.status == mapscript.MS_ON
        assert mock_layer.type == mapscript.MS_LAYER_POINT
        assert mock_layer.data == "the_geom from points"
        mock_layer.setProjection.assert_called_once_with("init=epsg:4326")
        mock_mapscript_map.insertLayer.assert_called_once_with(mock_layer)


@patch('mapscript.layerObj')
def test_add_vector_layer_linestring(mock_layer_obj, mock_mapscript_map):
    """Test adding a linestring vector layer to the map."""
    mock_layer = MagicMock()
    mock_layer_obj.return_value = mock_layer

    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()

        opts = gws.gis.ms.VectorLayerOptions(
            geometryType=gws.GeometryType.linestring,
            connectionType=None,
            connectionString=None,
            dataString="the_geom from lines",
            crs=gws.Data(srid=4326),
            style=None,
            config=None
        )

        layer = map_instance.add_vector_layer(opts)

        assert layer == mock_layer
        assert mock_layer.type == mapscript.MS_LAYER_LINE


@patch('mapscript.layerObj')
def test_add_vector_layer_polygon(mock_layer_obj, mock_mapscript_map):
    """Test adding a polygon vector layer to the map."""
    mock_layer = MagicMock()
    mock_layer_obj.return_value = mock_layer

    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()

        opts = gws.gis.ms.VectorLayerOptions(
            geometryType=gws.GeometryType.polygon,
            connectionType=None,
            connectionString=None,
            dataString="the_geom from polygons",
            crs=gws.Data(srid=4326),
            style=None,
            config=None
        )

        layer = map_instance.add_vector_layer(opts)

        assert layer == mock_layer
        assert mock_layer.type == mapscript.MS_LAYER_POLYGON


@patch('mapscript.layerObj')
def test_add_vector_layer_with_postgres(mock_layer_obj, mock_mapscript_map):
    """Test adding a vector layer with postgres connection type."""
    mock_layer = MagicMock()
    mock_layer_obj.return_value = mock_layer

    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()

        opts = gws.gis.ms.VectorLayerOptions(
            geometryType=gws.GeometryType.point,
            connectionType="postgres",
            connectionString="host=localhost dbname=test user=postgres",
            dataString="the_geom from points",
            crs=gws.Data(srid=4326),
            style=None,
            config=None
        )

        with pytest.raises(gws.Error, match="Invalid connectionType 'postgres'"):
            map_instance.add_vector_layer(opts)

        mock_layer.setConnectionType.assert_called_once_with(mapscript.MS_POSTGIS, '')


@patch('mapscript.layerObj')
def test_add_vector_layer_with_config(mock_layer_obj, mock_mapscript_map):
    """Test adding a vector layer with custom config."""
    mock_layer = MagicMock()
    mock_layer_obj.return_value = mock_layer

    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()

        opts = gws.gis.ms.VectorLayerOptions(
            geometryType=gws.GeometryType.point,
            connectionType=None,
            connectionString=None,
            dataString="the_geom from points",
            crs=gws.Data(srid=4326),
            style=None,
            config="LAYER\nCLASS\nEND\nEND"
        )

        layer = map_instance.add_vector_layer(opts)

        assert layer == mock_layer
        mock_layer.updateFromString.assert_called_once_with("LAYER\nCLASS\nEND\nEND")


@patch('gws.lib.image.from_bytes')
def test_map_draw(mock_from_bytes, mock_mapscript_map):
    """Test drawing a map."""
    mock_image = MagicMock()
    mock_from_bytes.return_value = mock_image

    mock_result = MagicMock()
    mock_result.getBytes.return_value = b"image data"
    mock_mapscript_map.draw.return_value = mock_result

    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()

        bounds = gws.Bounds(
            extent=[0, 0, 100, 100],
            crs=gws.Data(epsg="EPSG:4326", srid=4326)
        )
        size = gws.Size((800, 600))

        result = map_instance.draw(bounds, size)

        assert result == mock_image
        mock_mapscript_map.setExtent.assert_called_once_with(0, 0, 100, 100)
        mock_mapscript_map.setSize.assert_called_once_with(800, 600)
        mock_mapscript_map.setProjection.assert_called_once_with("EPSG:4326")
        mock_mapscript_map.draw.assert_called_once()
        mock_from_bytes.assert_called_once_with(b"image data")


def test_map_to_string(mock_mapscript_map):
    """Test converting a map to string."""
    with patch('mapscript.mapObj', return_value=mock_mapscript_map):
        map_instance = gws.gis.ms.Map()

        result = map_instance.to_string()

        assert result == "MAP\nEND"
        mock_mapscript_map.convertToString.assert_called_once()