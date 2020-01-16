Layer
======

Ein *layer* in einem GBD WebSuite Projekt wird durch seinen ``type`` identifiziert, zusätzlich haben Layer die folgenden Eigenschaften (wenn nicht explizit konfiguriert, wird er von der übergeordneten Ebene oder von der Map geerbt):

* ``source`` - wo die Ebene ihre Geodaten herbekommt (siehe :doc:`sources`)
* ``view`` - räumliche Eigenschaften der Ebene (Umfang, erlaubte Auflösungen oder Skalen für diese Ebene)
* ``cache`` und ``grid`` - beeinflussen das Layer-Caching (siehe :doc:`cache`)
* ``clientOptions`` - Optionen für den GBD WebSuite Client (siehe :doc:`client`)
* ``attribute`` - Layer-Metadaten (z. B. Attribution)
* ``meta`` - Transfomationsregeln für Features (siehe :doc:`features`)

Layer Typen
-------------

Box
~~~

Ein Box-Layer ist vergleichbar mit einem konventionellen WMS-Layer. Er wird mit den WMS-Parametern ``bbox``, ``width`` und ``height`` abgefragt und gibt ein ``png`` Bild zurück.


wms
~~~

Sie können festlegen, welche Layer Sie verwenden möchten. Alle WMS-Layer werden neu projiziert, kombiniert und als einzelnes Bild angezeigt:
    {
        "type": "wms",
        "title": "Webatlas.de",
        "sourceLayers": {
            "names": ["dtk250"]
        },
        "url": "http://sg.geodatenzentrum.de/wms_dtk250"
    }


wmts
~~~~

Wenn der Dienst mehrere Layer bereitstellt, können Sie angeben, welcher Layer verwendet werden soll:

    {
        "type": "wmts",
        "title": "NRW geobasis",
        "sourceLayer": "DOP",
        "url": "https://www.wmts.nrw.de/geobasis/wmts_nw_dop"
    }


Kachel
~~~~~~

Ein Kachel-Layer arbeitet als XYZ-Kachelquelle. Beachten Sie, dass in Abweichung von der allgemeinen Regel, Anfragen an Tile-Layer statische Anfragen imitieren, um clientseitiges Caching zu ermöglichen. Ein Beispiel für die Anforderung eines Kachellayer ::

    http://example.org/_/cmd/mapHttpGetXyz/layer/project.layer/z/1/x/2/y/3/t.png


Gruppe
~~~~~~~

Gruppenlayer enthalten andere Layer, sie liefern selbst keine Geodaten. Neben der visuellen Gruppierung besteht ein weiterer Zweck einer Gruppe darin, die Zugriffs- oder Fallback-Cache- und Grid-Konfigurationen für ihre untergeordneten Layer beizubehalten. Eine Gruppe kann "virtuell" oder ``unfolded`` erstellt werden, in diesem Fall wird sie im Client nicht angezeigt, während ihre untergeordneten Layer vorhanden sind. ::


Baum
~~~~

Ein Baumlayer ist in der Lage, eine ganze Hierarchie von Layern aus einer WMS- oder QGIS-Quelle darzustellen. Ein Baumlayer wird als Gruppe im Client und mit Quellen als Unterknoten (oder *leaves*) angezeigt.

Es ist auch möglich, nur bestimmte Layer aus der Quelle auszuwählen. Beim Lesen der Quelle erzeugt der Server eine virtuelle *path*-Eigenschaft für jeden Layer, die die eindeutige ID des Layers und ihre übergeordneten ids enthält, ähnlich den Pfaden des Dateisystems, wie ``/root-layer-id/grandparent-id/parent-id/layer-id``. Das ``pathMatch`` regex kann verwendet werden, um Layer mit passenden Pfaden zu filtern. ::


QGIS
~~~~

QGIS-Layer sind ähnlich wie Baumlayer, funktionieren aber nur mit QGIS-Karten. Anstelle eines einzelnen ``pathMatch`` können sie eine Liste von Matchregeln haben, die dem Server sagen, wie er mit passenden QGIS-Layern umgehen soll. Sie können z. B. einen bestimmten Layer "tilify oder einen bestimmten Teilbaum zu einem Layer "flatten".

QGIS-Layer zeigen ganze QGIS-Projekte als einzelne Gruppe im GWS-Layerbaum an. Zusätzlich zu einem Layerfilter können Sie angeben, ob entfernte (z. B. WMS-) Layer direkt gerendert und / oder durchsucht werden sollen, oder den QGIS-Server verwenden: ::

    {
        "type": "qgis",
        "title": "My qgis project",
        "path": "/data/path/to/my-project.qgis",
        "directRender": ["wms"]
    }


qgisflat
~~~~~~~

QGIS / WMS-Layer zeigen einzelne Layer aus einem QGIS-Projekt als einzelnes flaches Bild an ::
    {
        "type": "qgisflat",
        "title": "My qgis project",
        "path": "/data/path/to/my-project.qgis",
        "sourceLayers": [
            "names": ["My First Layer", "My Second Layer"]
        ]
    }


Vektor
~~~~~~~

Vektorlayer werden auf dem GBD WebSuite Client gerendert. Wenn ein Vektorlayer angefordert wird, sendet der Server die GeoJSON-Liste der Features und Stilbeschreibungen an den Client, der dann das eigentliche Rendering durchführt.


SQL
~~~

SQL-Layer übernehmen Geometrien aus einer SQL-Tabelle. Sie müssen nur den Datenbankanbieter und die zu verwendende Tabelle angeben ::
        {
            "title": "Houses",
            "type": "sql",
            "table": {
                "name": "myschema.mytable",
                "keyColumn": "id",
                "geometryColumn": "geom"
            },
            "loadingStrategy": "bbox",
            "style": {
                "type": "css",
                "text": "stroke:rgb(0,255,0); stroke-width:1px; fill:rgba(0,100,0,0.2)"
            }
        }
