# Layer :/admin-de/themen/layer

Layer sind die grundlegenden Bausteine einer Karte in der GBD WebSuite. Jeder Layer stellt eine Datenquelle dar und definiert, wie die darin enthaltenen Geodaten dargestellt werden. Layer werden innerhalb des `map`-Blocks eines Projekts konfiguriert und in der angegebenen Reihenfolge gerendert – der erste Eintrag liegt dabei ganz oben.

```javascript
{
    map {
        layers+ {
            type "wms"
            title "Hintergrundkarte"
            ...
        }
        layers+ {
            type "postgres"
            title "Fachdaten"
            ...
        }
    }
}

```

Alle Layer-Typen teilen sich eine gemeinsame Basis-Konfiguration:

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.base.layer.core.Config">gws.base.layer.core.Config</a>
</div>

## Raster-Layers

Raster-Layer liefern Geodaten als Rasterbild (PNG, JPEG). Die GBD WebSuite fragt dabei vom jeweiligen Dienst fertig gerenderte Kacheln oder Bilder ab und leitet diese an den Client weiter. Raster-Layer eignen sich besonders als Hintergrundkarten oder für Daten, die nur zur Visualisierung und nicht zur Abfrage vorgesehen sind.

### QGIS

Der `qgis`-Layer bindet ein QGIS-Projekt als Kartenquelle ein. Die GBD WebSuite kommuniziert dabei über den QGIS Server und übernimmt die im QGIS-Projekt definierte Layer-Hierarchie und Darstellung.

```javascript
{
    map {
        layers+ {
            type "qgis"
            title "Mein QGIS-Projekt"
            provider.path "/data/projekte/meinprojekt.qgs"
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.plugin.qgis.layer.Config">gws.plugin.qgis.layer.Config</a>
</div>

Ausführliche Informationen zur QGIS-Integration finden Sie unter [QGIS](/admin-de/plugin/qgis "QGIS").

### WMS

Ein `wms`-Layer bindet einen externen WMS-Dienst (Web Map Service) ein. Die Layer-Struktur des Dienstes wird dabei als Baum in der GBD WebSuite abgebildet.

```javascript
{
    map {
        layers+ {
            type "wms"
            title "Topographische Karte"
            provider.url "https://www.wms.nrw.de/geobasis/wms_nw_dtk"
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.plugin.ows_client.wms.layer.Config">gws.plugin.ows_client.wms.layer.Config</a>
</div>

### WMTS

Ein `wmts`-Layer bindet einen WMTS-Dienst (Web Map Tile Service) ein. WMTS-Dienste liefern vorberechnete Kacheln und sind daher besonders performant.

```javascript
{
    map {
        layers+ {
            type "wmts"
            title "Basiskarte"
            provider.url "https://basemap.de/wmts/1.0.0/WMTSCapabilities.xml"
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.plugin.ows_client.wmts.layer.Config">gws.plugin.ows_client.wmts.layer.Config</a>
</div>

### Tile

Der `tile`-Layer dient zur Einbindung von XYZ/Slippy-Map-Kacheldiensten (z. B. OpenStreetMap). Die Kachel-URL wird mit den Platzhaltern `{x}`, `{y}` und `{z}` angegeben.

```javascript
{
    map {
        layers+ {
            type "tile"
            title "OpenStreetMap"
            provider.url "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.plugin.tile_layer.layer.Config">gws.plugin.tile_layer.layer.Config</a>
</div>

### Kachel-Cache

Für Raster-Layer, die von externen Diensten abgefragt werden, kann ein lokaler Kachel-Cache aktiviert werden. Dadurch werden bereits abgerufene Kacheln zwischengespeichert und müssen bei erneuter Anfrage nicht erneut vom Dienst geladen werden.

```javascript
{
    map {
        layers+ {
            type "wms"
            title "Hintergrundkarte"
            provider.url "https://..."
            cache {
                maxAge "7d"
                maxLevel 14
            }
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.base.layer.core.CacheConfig">gws.base.layer.core.CacheConfig</a>
</div>

## Vector-Layers

Vektor-Layer liefern Geodaten als einzelne Objekte (Features) mit Geometrie und Attributen. Im Gegensatz zu Raster-Layern rendert die GBD WebSuite Vektor-Layer clientseitig und ermöglicht so Abfragen, Highlights und das Editieren einzelner Objekte.

### GeoJSON

Der `geojson`-Layer liest Vektordaten aus einer lokalen GeoJSON-Datei.

```javascript
{
    map {
        layers+ {
            type "geojson"
            title "Stadtteile"
            provider.path "/data/stadtteile.geojson"
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.plugin.geojson.layer.Config">gws.plugin.geojson.layer.Config</a>
</div>

### PostgreSQL

Der `postgres`-Layer liest Vektordaten direkt aus einer PostGIS-Tabelle. Er unterstützt Abfragen, Editieren und die Verwendung von Datenmodellen.

```javascript
{
    map {
        layers+ {
            type "postgres"
            title "Grundstücke"
            tableName "public.flurstuecke"
            dbUid "meine_datenbank"
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.plugin.postgres.layer.Config">gws.plugin.postgres.layer.Config</a>
</div>

Weitere Informationen zur Datenbankanbindung finden Sie unter [PostgreSQL](/admin-de/themen/postgresql "PostgreSQL"). Informationen zum Editieren von Objekten im Client finden Sie unter [Editieren](/admin-de/themen/edit "Editieren").

### WFS

Ein `wfs`-Layer bindet einen externen WFS-Dienst (Web Feature Service) ein und stellt dessen Objekte als Vektor-Features dar.

```javascript
{
    map {
        layers+ {
            type "wfs"
            title "Gewässer"
            provider.url "https://www.wfs.nrw.de/geobasis/wfs_nw_gewaesser"
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.plugin.ows_client.wfs.layer.Config">gws.plugin.ows_client.wfs.layer.Config</a>
</div>

### Styling

Die Darstellung von Vektor-Layern lässt sich über CSS-Selektoren und Stilregeln anpassen. Die Eigenschaft `cssSelector` ermöglicht es, auf einzelne Feature-Klassen gezielt Styles anzuwenden.

```javascript
{
    map {
        layers+ {
            type "postgres"
            title "Gebäude"
            tableName "public.gebaeude"
            cssSelector ".myLayer"
        }
    }
}

```

## Flat/Tree/Group Layers

Die GBD WebSuite unterscheidet zwischen **Tree-Layern**, **Flat-Layern** und **Gruppen-Layern**. Diese Konzepte bestimmen, wie Layer-Hierarchien aus externen Quellen im Client dargestellt werden.

### Tree-Layer und Flat-Layer

Einige Layer-Typen haben eine **Tree-Variante** und eine **Flat-Variante**:

| Typ | Variante | Beschreibung |
| --- | --- | --- |
| `qgis` | Tree | Übernimmt die Layer-Hierarchie des QGIS-Projekts als Baum |
| `qgisflat` | Flat | Stellt alle QGIS-Layer auf einer einzigen Ebene dar |
| `wms` | Tree | Übernimmt die Layer-Hierarchie des WMS-Dienstes als Baum |
| `wmsflat` | Flat | Alle WMS-Layer werden auf einer Ebene zusammengefasst |
| `wfs` | Tree | Übernimmt die Layer-Hierarchie des WFS-Dienstes als Baum |
| `wfsflat` | Flat | Alle WFS-Layer werden auf einer Ebene zusammengefasst |

Tree-Layer eignen sich, wenn die Quelllayer-Struktur im Client sichtbar und steuerbar sein soll. Flat-Layer sind praktisch, wenn alle Sublayer als eine einzige Einheit behandelt werden sollen.

<div class="admonition_info">Bei <code>qgisflat</code> und <code>wmsflat</code> ist kein einzelnes An- und Ausschalten von Sublayern möglich. Der gesamte Layer wird als Einheit dargestellt.
</div>

### Gruppen-Layer

Mit dem `group`-Layer können mehrere Layer zu einer logischen Einheit zusammengefasst werden. Im Client erscheint die Gruppe als ein Eintrag im Ebenenbaum, der auf- und zugeklappt werden kann.

```javascript
{
    map {
        layers+ {
            type "group"
            title "Hintergrundkarten"
            layers+ {
                type "wmts"
                title "Basiskarte farbig"
                ...
            }
            layers+ {
                type "wmts"
                title "Basiskarte grau"
                ...
            }
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.base.layer.group.Config">gws.base.layer.group.Config</a>
</div>

### Client-Optionen

Mit `clientOptions` lässt sich das Verhalten eines Layers im Client steuern:

| Eigenschaft | Typ | Beschreibung |
| --- | --- | --- |
| `hidden` | `bool` | Layer beim Start ausblenden |
| `selected` | `bool` | Layer beim Start aktiviert anzeigen |
| `expanded` | `bool` | Gruppe beim Start aufgeklappt anzeigen |
| `exclusive` | `bool` | Nur ein Sublayer der Gruppe kann gleichzeitig sichtbar sein |
| `unfolded` | `bool` | Gruppe selbst ausblenden, nur Sublayer anzeigen |
| `unlisted` | `bool` | Layer vollständig aus dem Ebenenbaum ausblenden |

```javascript
{
    map {
        layers+ {
            type "group"
            title "Hintergrundkarten"
            clientOptions {
                exclusive true
                expanded true
            }
            layers+ { ... }
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.base.layer.core.ClientOptions">gws.base.layer.core.ClientOptions</a>
</div>

### autoLayers

Bei Tree-Layern (`qgis`, `wms`, `wfs`) werden die Sublayer automatisch aus der Quelle generiert. Mit `autoLayers` können für diese automatisch erzeugten Sublayer Standardwerte überschrieben werden – etwa um bestimmten Layern andere `clientOptions` zu geben, ohne jeden Sublayer einzeln konfigurieren zu müssen.

```javascript
{
    map {
        layers+ {
            type "qgis"
            title "Stadtplan"
            provider.path "/data/stadtplan.qgs"
            autoLayers+ {
                applyTo.titles ["Hintergrund"]
                config.clientOptions.hidden true
            }
        }
    }
}

```

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.base.layer.core.AutoLayersOptions">gws.base.layer.core.AutoLayersOptions</a>
</div>
