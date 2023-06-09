# Layer :/admin-de/config-az/layer

Ein *Layer* in einem GBD WebSuite Projekt wird durch seinen ``type`` identifiziert. Grundsätzlich unterteilen sich Layer in  *Raster-* und *Vektorlayern*. Ein Rasterlayer stellt Geoinformation in der Pixelform dar. Die Rasterlayer enthalten keine Sachdaten (Attribute), sie können jedoch mit Suchquellen kombiniert werden, die Sachdaten enthalten. Ein Vektorlayer ist eine Sammlung von *Features*, aus geometrischen Objekten (Punkte, Linien und Polygone), zu denen auch Sachdaten zugeordnet werden. Die Darstellung von Vektorlayern können Sie mit Styling (siehe [Styling](/admin-de/config-az/styling)) frei gestalten.

## Layer ``uid``

Damit die Layer in dem System eindeutig identifiziert werden können, sind die Layer-IDs (``uid``) immer mit dem Projekt- und Map-``uid`` anzugeben (die Map-``uid`` ist immer ``map``). Zum Beispiel, wenn Sie diese Konfiguration haben:

```javascript

{
    "projects": [
        {
            "uid": "meinprojekt",
            "map": {
                "layers": [
                    {
                        "uid": "strassen"
                        ...
                    }
                ]
            }
        }
    ]
}
```

und Sie auf den Layer ``strassen`` woanders verweisen wollen (z.B. für Einrasten Funktion in [Bemaßung](/admin-de/config-az/bemassung)), dann ist die ``uid`` wie folgt anzugeben:

```javascript

{
    "type": "dimension",
    "layers": ["meinprojekt.map.strassen"]
    ...
}
```

Dies betrifft auch die URLs die das System z.B. für Map-Darstellung verwendet, z.B.

    http://example.com/_?cmd=mapHttpGetBox&projectUid=meinprojekt&layerUid=meinprojekt.map.strassen

## Konfiguration

%reference_de 'gws.ext.config.layer'

Bei einer Layerkonfiguration stehen viele Optionen zur Verfügung. Die wichtigsten davon sind:

|OPTION|BEDEUTUNG|
|---|---|
|``clientOptions`` | Optionen für darstellung im Client, siehe [client](/admin-de/config-az/client) |
|``display`` | Anzeige-Modus (``box`` - den ganzen Kartenausschnitt rendern, ``tile`` - gekachelte Darstellung ``client`` - Darstellung im Client mit Javascript) |
|``legend`` | Optionen für Legenden |
|``meta`` | Layer-Metadaten [Metadaten](/admin-de/config-az/metadaten) |
|``opacity`` | Transparenz als eine Fließkommazahl |
|``ows`` |  Eigenschaften im Bezug auf OWS Dienste, siehe [OWS](/admin-de/config-az/ows) |
|``search`` | Such-Provider, siehe [Suche](/admin-de/config-az/suche) |
|``templates`` | Vorlagen |
|``title`` | Layer-Titel |

Außerdem können Sie mit ``extent``, ``extentBuffer`` und ``zoom`` die entsprechenden Eigenschaften der Karte (siehe [Karte](/admin-de/config-az/karten)) überschreiben.

Für Vektorlayer sind zusätzlich diese Optionen vorhanden:

|OPTION|BEDEUTUNG|
|---|---|
|``dataModel`` | Feature Transformationsregeln, siehe [Feature](/admin-de/config-az/feature)|
|``edit`` und ``editDataModel`` | unterstützt Editieren von Layern, siehe [Digitalisierung](/admin-de/plugin/digitalisierung)|
|``loadingStrategy`` | steuert das Laden von Features (``all`` - alle laden, ``bbox`` - nur Features im Kartenausschnitt laden)|
|``style`` | Style für Features, siehe [Styling](/admin-de/config-az/styling)|

Für Rasterlayer können Sie mit ``cache`` und ``grid`` das Cachen von Rasterdaten steuern, siehe [Caching-Framework](/admin-de/config-az/cache).

## Quell-Layer

Die Layer die an externe Dienste angebunden sind (``wms``, ``wfs``, ``wmts``) sowie QGIS Layer bestehen normalerweise aus mehreren Quell-Layer. Jeder Quell-Layer, unabhängig von der Art der Quelle, hat in GWS folgende Eigenschaften:

|OPTION|BEDEUTUNG|
|---|---|
|``name`` | Name, entspricht ``<Layer><Name>`` in WMS und ``<FeatureType><Name>`` in WFS|
|``uid`` | Die ID wird aus dem Namen generiert, z.B. aus dem Namen "Bäume" entsteht die ID "baeume"|
|``path`` | Layer Pfad, die ID von Überlayer, mit einem Punkt getrennt|
|``level`` | Die Tiefe des Layers, wobei der Root-Layer Tiefe ``1`` hat|


Zum Beispiel, wenn ein WMS Dienst folgende Struktur hat:

```xml

    <Layer>
        <Name>Natur</Name>
        ...
        <Layer>
            <Name>Forst</Name>
            ...
            <Layer>
                <Name>Bäume</Name>
```

entstehen in GWS folgende Quell-Layer:

|``name`` | ``uid`` | ``path`` | ``level`` |
|---|---|---|---|
|``Natur`` | ``natur`` | ``natur`` | 1 |
|``Forst`` | ``forst`` | ``natur.forst`` | 2 |
|``Bäume`` | ``baeume`` | ``natur.forst.baeume`` | 3 |


Wenn Sie nur bestimmte Quell-Layer in Ihrem Projekt zeigen wollen, können Sie mit der Option ``sourceLayers`` die Quell-Layers nach Namen (``names``), Pfaden (``pattern``) oder Tiefe (``level``) filtern.

## Vorlagen

Für einen Layer können Sie eine Vorlage mit dem ``subject`` ``layer.description`` definieren, die im Client gezeigt wird, wenn der Nutzer den Layer auswählt. Bei Vektorlayern können zusätzlich Feature-Vorlagen definiert werden, siehe [Feature](/admin-de/config-az/feature).

## Legenden

%reference_de 'gws.ext.config.legend'

Mit der Option ``legend`` können Sie die Legende für den Layer konfigurieren. 

Sie haben die Wahl zwischen:

- einee *HTML* Vorlage für komplexe Legenden
- einem Bild als *statischer* Legende
- einer *Remote* Legende, abrufbar über eine URL
- einer *kombinierten* Legende aus mehreren Layern
- einer aus einem *QGIS*-Provider generierten Legende


## Layer Typen

### geojson

%reference_de 'gws.plugin.geojson.layer.Config'

Vektorlayer, der die Daten aus einer [GeoJSON](https://geojson.org/) Datei darstellt.

### group

%reference_de 'gws.base.layer.group.Config'

Gruppenlayer enthalten andere Layer, sie liefern selbst keine Geodaten. Neben der visuellen Gruppierung besteht ein weiterer Zweck einer Gruppe darin, die Zugriffs- bzw Ausmaß-Konfigurationen für ihre untergeordneten Layer beizubehalten. Eine Gruppe kann "virtuell" oder ``unfolded`` erstellt werden, in diesem Fall wird sie im Client nicht angezeigt, während ihre untergeordneten Layer vorhanden sind.

### postgres

%reference_de 'gws.plugin.postgres.layer.Config'

Vektor-Layer, übernimmt Geometrien aus einer Postgres-Tabelle. Sie müssen die zu verwendende Tabelle angeben

```javascript

{
    "title": "Houses",
    "type": "postgres",
    "tableName": "myschema.mytable",
    "loadingStrategy": "bbox"
}
```

Mehr über Tabellen-Konfigurationen siehe [Datenbanken](/admin-de/config-az/datenbanken).

### qgis/qgisflat

%reference_de  'gws.plugin.qgis.qgisflat_layer.Config'

QGIS Layer, mehr dazu in [QGIS](/admin-de/config-az/layer).

### tile

%reference_de 'gws.plugin.tile_layer.layer.Config'

Rasterlayer, zeigt Kachel aus einer XYZ-Kachelquelle (wie z.B. OSM). Sie müssen die URL mit Platzhaltern ``{{{x}}}``, ``{{{y}}}`` und ``{{{z}}}`` angeben

```javascript

{
    "type": "title",
    "title": "Open Street Map",
    "display": "tile",
    "provider": {"url": "https://a.tile.openstreetmap.org/{{{z}}}/{{{x}}}/{{{y}}}.png"}
}
```

### wfs

%reference_de 'gws.plugin.ows_provider.wfs.wfs_layer.Config'

Vektorlayer, zeigt Features aus einen WFS Dienst

```javascript

{
    "type": "wfs",
    "title": "Geobasis NRW WFS Service",
    "url": "https://www.wfs.nrw.de/geobasis/wfs_nw_dvg",
    "sourceLayers": {
        "pattern": "nw_dvg1_gem"
    }
}
```

### wms

%reference_de 'gws.plugin.ows_provider.wms.wms_layer.Config'

Rasterlayer, zeigt Rasterdaten aus einem WMS Dienst. Falls der Dienst mehrere Layer enthält, werden diese als eine Baumstruktur dargestellt

```javascript

{
    "type": "wms",
    "title": "Webatlas.de - Alle Layer",
    "url": "http://sg.geodatenzentrum.de/wms_dtk250"
}
```

### wmsflat

%reference_de 'gws.plugin.ows_provider.wfs.wfsflat_layer.Config'

Rasterlayer, zeigt Rasterdaten aus einem WMS Dienst. Die WMS-Layer werden kombiniert, ggf. umprojiziert,  und als einzelnes Bild angezeigt

```javascript

{
    "type": "wmsflat",
    "title": "Webatlas.de - DTK250",
    "sourceLayers": {
        "names": ["dtk250"]
    },
    "url": "http://sg.geodatenzentrum.de/wms_dtk250"
}
```

### wmts

%reference_de 'gws.plugin.ows_provider.wmts.layer.Config'

Rasterlayer, zeigt Rasterdaten aus einem WMTS Dienst

```javascript

{
    "type": "wmts",
    "title": "Geobasis NRW WMTS Service",
    "sourceLayer": "DOP",
    "url": "https://www.wmts.nrw.de/geobasis/wmts_nw_dop"
}
```