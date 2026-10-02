# OWS-Dienst "wmts" :/admin-de/konfiguration/owsService/wmts

Der Dienst `wmts` stellt die Layer eines Projekts als Web Map Tile Service bereit, sodass externe Anwendungen vorberechnete Kartenkacheln abrufen können. Unterstützt wird die WMTS-Version 1.0.0; verarbeitet werden ausschließlich `GET`-Anfragen mit KVP-Kodierung.

## Vorlagen

Der Dienst unterstützt eine Vorlage:

| Subject | Erzeugtes Dokument |
|---|---|
| `ows.GetCapabilities` | Capabilities-Dokument des Dienstes |

`GetTile` und `GetLegendGraphic` liefern Bilder und lassen sich nicht über Vorlagen anpassen; ihr Ausgabeformat bestimmen die Bildformate des Dienstes.

## Beispiel-Konfiguration ::

```javascript
owsServices+ {
    type "wmts"
    supportedCrs [ 3857 25832 ]
    metadata {
        name "MEINE_STADT_WMTS"
        title "WMTS Meine Stadt"
    }
}

map.extent [ 723753, 6631615, 783753, 6691615 ]

map.layers+ {
    title "Basiskarte"
    type "qgis"
    provider.path "/data/qgis/basiskarte.qgs"
    withCache true
}
```

Der Dienst liefert vorberechnete Kacheln der Projektkarte. Für jedes in `supportedCrs` angegebene Koordinatenbezugssystem wird ein eigener `TileMatrixSet` erzeugt; der `map.extent` – hier in EPSG:3857 als erstem unterstützten CRS – begrenzt den Bereich der Kachelmatrix. Über `map.layers` legen Sie den Karteninhalt fest; `withCache true` bewirkt, dass die Kacheln zwischengespeichert und nicht bei jeder Anfrage neu gerendert werden.

%ref "gws.plugin.ows_server.wmts.Config"
%demo "service_wmts_qgis_demo"
%demo "service_wmts_raster_demo"
%demo "service_wmts_simple_demo"
