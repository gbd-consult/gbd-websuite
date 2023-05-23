# Qgis :/admin-de/plugin/qgis

## Layer

Es gibt zwei Möglichkeiten, QGIS Projekte als Layer in Ihren Karten einzubinden: ``qgis`` und ``qgisflat`` Layer. Sie können diese Layers frei mit anderen Layertypen mischen.

## qgis

TODO! %reference_de 'gws.ext.layer.qgis.Config'

QGIS-Layer zeigen ganze QGIS-Projekte als einzelne Gruppe im GWS-Layerbaum an. Zusätzlich zu einem Layerfilter können Sie angeben, ob entfernte (z. B. WMS-) Layer direkt gerendert und / oder durchsucht werden sollen, oder den QGIS-Server verwenden:

```javascript

{
    "type": "qgis",
    "title": "My qgis project",
    "path": "/data/path/to/my-project.qgis",
    "directRender": ["wms"]
}
```

Sie können auch die Struktur der Gruppe mit ``flattenLayers`` beinflüssen, indem Sie z.B. nur die Layers bis Tiefe 2 Zeigen und tiefere Layers flach darstellen.

## qgisflat

TODO! %reference_de 'gws.ext.layer.qgisflat.Config'

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

Für QGIS Layer können Sie unter ``legend.options`` einige visuelle Einstellungen für Legenden anpassen. Eine Liste der möglichen Einstellungen finden Sie in der `QGIS Referenz <https://docs.qgis.org/testing/en/docs/server_manual/services.html#getlegendgraphics>`_.

## Druckvorlagen

TODO! %reference_de 'gws.ext.template.qgis.Config'

Sie können QGIS Druckvorlagen ("Layouts") frei verwenden, auch für nicht-QGIS bzw. gemischte Karten. In der Vorlagen-Konfiguration muss den Pfad zu der QGIS Projektdatei angegeben werden, sowie Layout-Namen bzw Nummer. In dem Layout wird das ``Map`` Element mit der aktuellen GWS Karte ersetzt, zusätzlich können Sie in ``HTML-Frame`` Elements einige ``gws:`` Tags nutzen, wie z.B. ``<gws:legend>`` (siehe [Drucken](/admin-de/config-az/print)). Die vom Nutzer definierte Druck-Attribute können Sie mit dem QGIS Syntax ``[% @variable %]`` einfügen.

%info
 Beachten Sie, dass der Hintergrund der Vorlage (unter "Seiteneigenschaften") transparent sein muss.
%end

## Server Einstelligen

%reference_de 'gws.server.core.QgisConfig'

In der Serverkonfiguration (siehe [Server](/admin-de/config-az/server)) gibt es einige Optionen, die die Werte von QGIS-Umgebungsvariablen setzen. Die genaue Bedeutung entnehmen Sie bitte der QGIS-Dokumentation:

| GWS Option | QGIS Umgebungsvariable |
| ``debug``	| ``QGIS_DEBUG`` |
| ``maxCacheLayers`` | ``MAX_CACHE_LAYERS`` |
| ``serverCacheSize`` | ``QGIS_SERVER_CACHE_SIZE`` |
| ``serverLogLevel`` | ``QGIS_SERVER_LOG_LEVEL`` |

Die Option ``searchPathsForSVG`` zeigt an, wo der Server SVG-Bilder in QGIS-Karten und Druckvorlagen findet. Wenn Sie nicht standardmäßige Bilder verwenden, fügen Sie einfach einen Verzeichnispfad für sie zu dieser Einstellung hinzu.
