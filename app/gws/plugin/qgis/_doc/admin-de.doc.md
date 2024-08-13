# QGIS :/admin-de/plugin/qgis

## Layer

Es gibt zwei Möglichkeiten, QGIS Projekte als Layer in Ihren Karten einzubinden: ``qgis`` und ``qgisflat`` Layer. Sie können diese Layer frei mit anderen Layertypen mischen.

## qgis

%reference_de 'gws.plugin.qgis.layer.Config'

QGIS-Layer zeigen ganze QGIS-Projekte als einzelne Gruppe im GWS-Layerbaum an. Zusätzlich zu einem Layerfilter können Sie angeben, ob entfernte (z. B. WMS-) Layer direkt gerendert und / oder durchsucht werden sollen, oder den QGIS-Server verwenden:

```javascript

{
    "type": "qgis",
    "title": "My qgis project",
    "path": "/data/path/to/my-project.qgis",
    "directRender": ["wms"]
}
```

Sie können auch die Struktur der Gruppe mit ``flattenLayers`` beinflussen, indem Sie sich z.B. nur die Layer bis zur Tiefe 2 anzeigen lassen und alle tieferen Layer flach darstellen.

## qgisflat

%reference_de 'gws.plugin.qgis.flatlayer.Config'

``qgisflat``-Layer zeigen einzelne Layer aus einem QGIS-Projekt als einzelnes flaches Bild an:

```javascript

{
    "type": "qgisflat",
    "title": "My qgis project",
    "path": "/data/path/to/my-project.qgis",
    "sourceLayers": [
        "names": ["My First Layer", "My Second Layer"]
    ]
}
```

## Legenden

Für QGIS Layer können Sie unter ``legend.options`` einige visuelle Einstellungen für Legenden anpassen. Eine Liste der möglichen Einstellungen finden Sie in der [QGIS Referenz](https://docs.qgis.org/testing/en/docs/server_manual/services.html#getlegendgraphics).

## Druckvorlagen

%reference_de 'gws.plugin.qgis.template.Config'

Sie können QGIS Druckvorlagen ("Layouts") frei verwenden, auch für nicht-QGIS bzw. gemischte Karten. In der Vorlagen-Konfiguration muss dabei der Pfad zu der QGIS Projektdatei, sowie Layout-Namen bzw Nummer angegeben werden. In dem Layout wird das ``Map`` Element mit der aktuellen GWS Karte ersetzt, zusätzlich können Sie in ``HTML-Frame`` Elements einige ``gws:`` Tags nutzen, wie z.B. ``<gws:legend>`` (siehe [Drucken](/admin-de/themen/printer)). Die vom Nutzer definierte Druck-Attribute können Sie mit dem QGIS Syntax ``[% @variable %]`` einfügen.

%info
 Beachten Sie, dass die Deckkraft des Hintergrunds der Druckvorlage auf 0% gesetzt sein muss (unter "Elementeigenschaften").
%end

## Server Einstellungen

%reference_de 'gws.server.core.QgisConfig'

In der Serverkonfiguration (siehe [Server](/admin-de/config/server)) gibt es einige Optionen, die die Werte von QGIS-Umgebungsvariablen setzen. Die genaue Bedeutung entnehmen Sie bitte der QGIS-Dokumentation:

| GWS Option | QGIS Umgebungsvariable |
| ``debug``	| ``QGIS_DEBUG`` |
| ``maxCacheLayers`` | ``MAX_CACHE_LAYERS`` |
| ``serverCacheSize`` | ``QGIS_SERVER_CACHE_SIZE`` |
| ``serverLogLevel`` | ``QGIS_SERVER_LOG_LEVEL`` |

Die Option ``searchPathsForSVG`` zeigt an, wo der Server SVG-Bilder in QGIS-Karten und Druckvorlagen findet. Wenn Sie nicht standardmäßige Bilder verwenden, fügen Sie einfach einen Verzeichnispfad für diese zu der Einstellung hinzu.
