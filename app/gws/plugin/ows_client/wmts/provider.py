"""WMTS service provider."""

from typing import Optional, cast

import gws
import gws.base.layer
import gws.base.ows.client
import gws.config.util
import gws.lib.uom
import gws.lib.net

from . import caps


class Config(gws.base.ows.client.provider.Config):
    """Connection to a WMTS service."""


class Object(gws.base.ows.client.provider.Object):
    """WMTS service provider.

    Reads the capabilities of a WMTS service, builds tile URL templates and
    fetches tiles for the layers that use the service.
    """

    protocol = gws.OwsProtocol.WMTS

    tileMatrixSets: list[gws.TileMatrixSet]
    """Tile matrix sets offered by the service."""

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = cc.metadata
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version
        self.tileMatrixSets = cc.tileMatrixSets

        self.configure_operations(cc.operations)

    def tile_url_template(self, sl: gws.SourceLayer, tms: gws.TileMatrixSet, style: gws.SourceStyle) -> str:
        """Create a tile URL template for a source layer.

        If the layer has a ``ResourceURL`` of type ``tile``, its template is used,
        with ``{TileMatrixSet}`` and ``{Style}`` filled in. Otherwise, a KVP GetTile
        URL is built, with the layer format or ``image/jpeg``. In both cases the
        result contains the ``{TileMatrix}``, ``{TileCol}`` and ``{TileRow}`` placeholders.

        Args:
            sl: Source layer.
            tms: Tile matrix set.
            style: Source style.

        Returns:
            The URL template.
        """
        ru = sl.resourceUrls
        resource_url = ru.get('tile') if ru else None

        if resource_url:
            return (
                resource_url
                .replace('{TileMatrixSet}', tms.identifier)
                .replace('{Style}', style.name))

        params = {
            'SERVICE': gws.OwsProtocol.WMTS,
            'REQUEST': gws.OwsVerb.GetTile,
            'VERSION': self.version,
            'LAYER': sl.name,
            'FORMAT': sl.imageFormat or 'image/jpeg',
            'TILEMATRIXSET': tms.identifier,
            'STYLE': style.name,
            'TILEMATRIX': '{TileMatrix}',
            'TILECOL': '{TileCol}',
            'TILEROW': '{TileRow}',
        }

        op = self.get_operation(gws.OwsVerb.GetTile)
        args = self.prepare_operation(op, params=params)
        url = gws.lib.net.add_params(args.url, args.params)

        # {} should not be encoded
        return url.replace('%7B', '{').replace('%7D', '}')

    def get_tile(self, url_template: str, matrix_uid: str, col: int, row: int) -> bytes:
        """Fetch a tile.

        Args:
            url_template: URL template from ``tile_url_template``.
            matrix_uid: Tile matrix identifier.
            col: Tile column.
            row: Tile row.

        Returns:
            The tile image content.

        Raises:
            ``gws.ExternalServiceError``: If the request fails or the response is not an image.
        """
        url = url_template
        url = url.replace('{TileMatrix}', matrix_uid)
        url = url.replace('{TileCol}', str(col))
        url = url.replace('{TileRow}', str(row))

        res = gws.lib.net.http_request(url)
        if not res.ok:
            raise gws.ExternalServiceError(f'tile request failed: status={res.status_code} url={url!r}')
        if not res.content_type.startswith('image/'):
            raise gws.ExternalServiceError(f'tile request failed: content type {res.content_type!r} url={url!r}')
        return res.content
