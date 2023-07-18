# Layer :/admin-de/config/layer

Ein *Layer* in einem GBD WebSuite Projekt wird durch seinen ``type`` identifiziert. Grundsätzlich unterteilen sich Layer in  *Raster-* und *Vektorlayern*. Ein Rasterlayer stellt Geoinformation in der Pixelform dar. Die Rasterlayer enthalten keine Sachdaten (Attribute), sie können jedoch mit Suchquellen kombiniert werden, die Sachdaten enthalten. Ein Vektorlayer ist eine Sammlung von *Features*, aus geometrischen Objekten (Punkte, Linien und Polygone), zu denen auch Sachdaten zugeordnet werden. Die Darstellung von Vektorlayern können Sie mit Styling (siehe [Styling](/admin-de/config/style)) frei gestalten.

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

und Sie auf den Layer ``strassen`` woanders verweisen wollen (z.B. für Einrasten Funktion in [Bemaßung](/admin-de/plugin/dimension)), dann ist die ``uid`` wie folgt anzugeben:

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
|``clientOptions`` | Optionen für darstellung im Client, siehe [client](/admin-de/config/client) |
|``display`` | Anzeige-Modus (``box`` - den ganzen Kartenausschnitt rendern, ``tile`` - gekachelte Darstellung ``client`` - Darstellung im Client mit Javascript) |
|``legend`` | Optionen für Legenden |
|``meta`` | Layer-Metadaten [Metadaten](/admin-de/config/metadaten) |
|``opacity`` | Transparenz als eine Fließkommazahl |
|``ows`` |  Eigenschaften im Bezug auf OWS Dienste, siehe [OWS](/admin-de/config/ows) |
|``search`` | Such-Provider, siehe [Suche](/admin-de/config/suche) |
|``templates`` | Vorlagen |
|``title`` | Layer-Titel |

Außerdem können Sie mit ``extent``, ``extentBuffer`` und ``zoom`` die entsprechenden Eigenschaften der Karte (siehe [Karte](/admin-de/config/karten)) überschreiben.

Für Vektorlayer sind zusätzlich diese Optionen vorhanden:

|OPTION|BEDEUTUNG|
|---|---|
| `model`                    | Feature Transformationsregeln, siehe [Model](/admin-de/config/model)|
| `edit` und `editDataModel` | unterstützt Editieren von Layern, siehe [Digitalisierung](/admin-de/plugin/edit) |
| `loadingStrategy`          | steuert das Laden von Features (``all`` - alle laden, ``bbox`` - nur Features im Kartenausschnitt laden)|
| `style`                    | Style für Features, siehe [Styling](/admin-de/config/style)|

Für Rasterlayer können Sie mit ``cache`` und ``grid`` das Cachen von Rasterdaten steuern, siehe [Caching-Framework](/admin-de/config/cache).

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

Für einen Layer können Sie eine Vorlage mit dem ``subject`` ``layer.description`` definieren, die im Client gezeigt wird, wenn der Nutzer den Layer auswählt. Bei Vektorlayern können zusätzlich Feature-Vorlagen definiert werden, siehe [Feature](/admin-de/config/feature).

## Legenden

%reference_de 'gws.ext.config.legend'

Mit der Option ``legend`` können Sie die Legende für den Layer konfigurieren. 

Sie haben die Wahl zwischen:

- keine Legende für den Layer zeigen. In dem Fall setzen Sie `enabled` auf `false`
- automatische Legende zeigen (z.B. für WMS Layer): setzen Sie `enabled` auf `true`
- ein Bild als Legende zeigen, geben Sie die `url` des Bildes an
- eine HTML Vorlage verwenden für eine komplexe Legende. Nutzen Sie die `template` Eigenschaft

## Layer Typen :/admin-de/config/layer/type

### :/admin-de/config/layer/type/geojson

### group

%reference_de 'gws.base.layer.group.Config'

Gruppenlayer enthalten andere Layer, sie liefern selbst keine Geodaten. Neben der visuellen Gruppierung besteht ein weiterer Zweck einer Gruppe darin, die Zugriffs- bzw Ausmaß-Konfigurationen für ihre untergeordneten Layer beizubehalten. Eine Gruppe kann "virtuell" oder ``unfolded`` erstellt werden, in diesem Fall wird sie im Client nicht angezeigt, während ihre untergeordneten Layer vorhanden sind.

### :/admin-de/config/layer/type/postgres

### qgis/qgisflat

%reference_de  'gws.plugin.qgis.qgisflat_layer.Config'

QGIS Layer, mehr dazu in [QGIS](/admin-de/config/layer).

### :/admin-de/config/layer/type/tile

### :/admin-de/config/layer/type/wfs

### :/admin-de/config/layer/type/wms

### :/admin-de/config/layer/type/wmsflat

### :/admin-de/config/layer/type/wmts


# Caching Framework :/admin-de/config/cache

Der GBD WebSuite Server kann Geo-Bilder aus externen Quellen auf der Festplatte zwischenspeichern (bzw. *cachen*), sodass weitere Anfragen zu der selben Quelle viel schneller ausgeführt werden können.  Das Cache Verzeichnis befindet sich in dem von Ihnen konfigurierten *var* Verzeichnis und kann bei Bedarf jederzeit komplett gelöscht werden

%info
 Die Caches können sehr viel Speicherplatz benötigen. Sorgen Sie dafür, dass Ihr Dateisystem über ausreichend freien Platz und freie **inodes** verfügt.
%end

## Layer Konfiguration

Das Cachen kann für jeden Layer mit den Optionen `cache` und `grid` flexibel konfiguriert werden.

### cache

%reference_de 'gws.base.layer.types.CacheConfig'

Geben Sie hier an, ob Caching aktiviert ist und für wie lange die gecachten Bilder gespeichert werden sollen.

### grid

%reference_de 'gws.base.layer.types.GridConfig'

Geben Sie hier an, wie der Kachelgrid für diesen Layer aussieht. Bei den Layern die keine Kachel liefern, wie `wms` oder `qgisflat` ist es wichtig einen ausreichenen Puffer (`reqBuffer`) zu setzen, damit die Beschriftungen richtig positioniert werden.

## Seeding

%reference_de 'gws.base.application.SeedingConfig'
^CLIREF cache.seed

Sobald der Cache eingerichtet ist, wird er automatisch gefüllt wenn Benutzer Ihre Karten in Browser anschauen. Sie können den Cache auch mit den Kommandozeilen-Tools `gws cache` befüllen (sogenanntes *Seeding*).

## Verwaltung von Cache

^CLIREF cache.clean

Mit dem selben Tool können Sie den Status des Cache abfragen oder individuelle Caches löschen.

%info
 Wenn Sie Ansichts- oder Rasterkonfigurationen ändern, müssen Sie den Cache für die Ebene oder die Karte entfernen, um unangenehme Artefakte zu vermeiden.
%end


