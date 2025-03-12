"""MapProxy integration module for GWS.

This module provides functions to interact with MapProxy services for WMS and WMTS requests.
"""

import time
from typing import Any, Dict, Optional, Union

import gws
import gws.config
import gws.lib.net


class ServiceException(Exception):
    """Exception raised when a MapProxy service request fails."""
    pass


def _call(service: str, params: Dict[str, Any]) -> bytes:
    """Make a direct call to a MapProxy service.

    Args:
        service: The service endpoint name.
        params: The parameters to send with the request.

    Returns:
        The image content as bytes if successful.

    Raises:
        ServiceException: If the response is not an image.
    """
    url = getattr(gws.config.get_root().app, 'mpxUrl') + '/' + service
    resp = gws.lib.net.http_request(url, params=params)
    if resp.content_type.startswith('image'):
        return resp.content
    raise ServiceException(resp.text)


_retry_count = 3
_retry_pause = 5
_request_number = 0


def _call_with_retry(service: str, params: Dict[str, Any]) -> Optional[bytes]:
    """Call a MapProxy service with retry logic.

    Makes multiple attempts to call the service if initial attempts fail.

    Args:
        service: The service endpoint name.
        params: The parameters to send with the request.

    Returns:
        The image content as bytes if successful, None if all retries fail.
    """
    global _request_number

    _request_number += 1
    rc = 0

    while True:
        try:
            return _call(service, params)
        except Exception as exc:
            gws.log.exception()
            err = repr(exc)

        if rc >= _retry_count:
            gws.log.error(f'MAPPROXY_ERROR: {_request_number}/{rc} FAILED error={err} {params!r}')
            return None

        gws.log.error(f'MAPPROXY_ERROR: {_request_number}/{rc} retry error={err}')
        time.sleep(_retry_pause)
        rc += 1


def wms_request(layer_uid: str, bounds: gws.Bounds, width: int, height: int, 
                forward: Optional[Dict[str, Any]] = None) -> Optional[bytes]:
    """Make a WMS GetMap request to MapProxy.

    Args:
        layer_uid: The layer identifier.
        bounds: The bounding box for the map.
        width: The width of the requested image in pixels.
        height: The height of the requested image in pixels.
        forward: Additional parameters to forward to the WMS request.

    Returns:
        The image content as bytes if successful, None if the request fails.
    """
    params = {
        'bbox': bounds.extent,
        'width': width,
        'height': height,
        'crs': bounds.crs.epsg,
        'service': 'WMS',
        'request': 'GetMap',
        'version': '1.3.0',
        'format': 'image/png',
        'transparent': 'true',
        'styles': '',
        'layers': layer_uid
    }
    if forward:
        params.update(forward)
    return _call_with_retry('wms', params)


def wmts_request(source_uid: str, x: int, y: int, z: int, 
                 tile_matrix: str, tile_size: int) -> Optional[bytes]:
    """Make a WMTS GetTile request to MapProxy.

    Args:
        source_uid: The source layer identifier.
        x: The tile column.
        y: The tile row.
        z: The zoom level.
        tile_matrix: The tile matrix set identifier.
        tile_size: The size of the tile in pixels.

    Returns:
        The tile image content as bytes if successful, None if the request fails.
    """
    params = {
        'tilecol': x,
        'tilerow': y,
        'tilematrix': z,
        'service': 'WMTS',
        'request': 'GetTile',
        'version': '1.0.0',
        'format': 'image/png',
        'tilematrixset': tile_matrix,
        'style': 'default',
        'layer': source_uid
    }
    return _call_with_retry('ows', params)
