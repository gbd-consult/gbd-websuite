Layer
=====

^REF gws.types.ext.layer.Config

Ein *layer* in einem GBD WebSuite Projekt wird durch seinen ``type`` identifiziert, zusätzlich haben Layer die folgenden Eigenschaften:

* ``source`` - wo die Ebene ihre Geodaten herbekommt (s. ^sources)
* ``view`` - räumliche Eigenschaften der Ebene (Umfang, erlaubte Auflösungen oder Skalen für diese Ebene)
* ``cache`` und ``grid`` - beeinflussen das Layer-Caching (s. ^cache)
* ``clientOptions`` - Optionen für den GBD WebSuite Client (s. ^client)
* ``attribute`` - Layer-Metadaten (z. B. Attribution)
* ``meta`` - Transfomationsregeln für Features (s. ^features)

Layer Typen
-----------

geojson
~~~~~~~

^REF gws.ext.layer.geojson.Config

group
~~~~~

^REF gws.ext.layer.group.Config

Gruppenlayer enthalten andere Layer, sie liefern selbst keine Geodaten. Neben der visuellen Gruppierung besteht ein weiterer Zweck einer Gruppe darin, die Zugriffs- oder Fallback-Cache- und Grid-Konfigurationen für ihre untergeordneten Layer beizubehalten. Eine Gruppe kann "virtuell" oder ``unfolded`` erstellt werden, in diesem Fall wird sie im Client nicht angezeigt, während ihre untergeordneten Layer vorhanden sind.

postgres
~~~~~~~~

^REF gws.ext.layer.postgres.Config

Postgres-Layer übernehmen Geometrien aus einer Postgres-Tabelle. Sie müssen nur den Datenbankanbieter und die zu verwendende Tabelle angeben ::

        {
            "title": "Houses",
            "type": "sql",
            "table": {
                "name": "myschema.mytable"
            },
            "loadingStrategy": "bbox",
            "style": {
                "type": "css",
                "text": "stroke:rgb(0,255,0); stroke-width:1px; fill:rgba(0,100,0,0.2)"
            }
        }

qgis
~~~~

^REF gws.ext.layer.qgis.Config

QGIS-Layer zeigen ganze QGIS-Projekte als einzelne Gruppe im GWS-Layerbaum an. Zusätzlich zu einem Layerfilter können Sie angeben, ob entfernte (z. B. WMS-) Layer direkt gerendert und / oder durchsucht werden sollen, oder den QGIS-Server verwenden: ::

    {
        "type": "qgis",
        "title": "My qgis project",
        "path": "/data/path/to/my-project.qgis",
        "directRender": ["wms"]
    }

qgisflat
~~~~~~~~

^REF gws.ext.layer.qgisflat.Config

``qgisflat``-Layer zeigen einzelne Layer aus einem QGIS-Projekt als einzelnes flaches Bild an ::

    {
        "type": "qgisflat",
        "title": "My qgis project",
        "path": "/data/path/to/my-project.qgis",
        "sourceLayers": [
            "names": ["My First Layer", "My Second Layer"]
        ]
    }

tile
~~~~

^REF gws.ext.layer.tile.Config

Ein Kachel-Layer arbeitet als XYZ-Kachelquelle. Beachten Sie, dass in Abweichung von der allgemeinen Regel, Anfragen an Tile-Layer statische Anfragen imitieren, um clientseitiges Caching zu ermöglichen. Ein Beispiel für die Anforderung eines Kachellayer ::

    http://example.org/_/cmd/mapHttpGetXyz/layer/project.layer/z/1/x/2/y/3/t.png

wfs
~~~

^REF gws.ext.layer.wfs.Config

wms
~~~

^REF gws.ext.layer.wms.Config

Sie können festlegen, welche Layer Sie verwenden möchten. Alle WMS-Layer werden neu projiziert, kombiniert und als einzelnes Bild angezeigt ::

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

^REF gws.ext.layer.wmts.Config

Wenn der Dienst mehrere Layer bereitstellt, können Sie angeben, welcher Layer verwendet werden soll ::

    {
        "type": "wmts",
        "title": "NRW geobasis",
        "sourceLayer": "DOP",
        "url": "https://www.wmts.nrw.de/geobasis/wmts_nw_dop"
    }

Ansichtsoptionen
----------------

Client-Optionen
---------------

Legenden
--------
