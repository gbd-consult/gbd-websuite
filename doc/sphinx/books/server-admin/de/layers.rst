Layer
======

Ein *Layer* in einem GBD WebSuite Projekt wird durch seinen ``type`` identifiziert. Zusätzlich haben die Layer folgende Eigenschaften. Falls diese nicht explizit konfiguriert sind, werden die Eigenschaften von der übergeordneten Ebene oder Map geerbt.

* ``source`` - die Quelle von der der Layer seine Geodaten bezieht (siehe :doc:`sources`)
* ``view`` - räumliche Eigenschaften des Layers (Umfang, erlaubte Auflösungen oder Zoomstufen für diesen Layer)
* ``cache`` und ``grid`` - konfigurieren des Layer-Cachings (siehe :doc:`cache`)
* ``clientOptions`` - Optionen für den GBD WebSuite Client (siehe :doc:`client`)
* ``attribute`` - Layer-Metadaten (z. B. Attribution)
* ``meta`` - Transfomationsregeln für Features (siehe :doc:`features`)

Layer Typen
-----------

Boxlayer
~~~~~~~~

Ein Box-Layer ist vergleichbar mit einem konventionellen WMS-Layer. Er wird mit den WMS-Parametern ``bbox``, ``width`` und ``height`` abgefragt und liefert ein ``png`` Bild.


Kachellayer
~~~~~~~~~~~

Ein Kachellayer generiert ein Bild aus einzelnen XYZ-Kacheln. Beachten Sie, dass in Abweichung von der allgemeinen Regel, Anfragen an Kachellayer, statische Anfragen imitieren, um clientseitiges Caching zu ermöglichen. Ein Beispiel für die Anforderungen eines Kachellayers ::

    http://example.org/_/cmd/mapHttpGetXyz/layer/project.layer/z/1/x/2/y/3/t.png

    {
        "type": "tile",
        "title": "Wikimaps",
        "url": "https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png"
    }

Gruppenlayer
~~~~~~~~~~~~

Gruppenlayer gruppieren verschiedene andere Layer. Sie selber liefern keine Geodaten sondern dienen lediglich der Sortierung und Ordnung. Neben der visuellen Gruppierung dient eine  Gruppe des Weiteren um die Zugriffs- oder Fallback-Cache- und Grid-Konfigurationen für ihre untergeordneten Layer beizubehalten. Eine Gruppe kann "virtuell" oder ``unfolded`` erstellt werden. Ist die Gruppe ``unfolded`` erstellt worden, wird sie in diesem Fall im Client nicht angezeigt, während ihre untergeordneten Layer jedoch angezeigt werden::

    {
        "type": "group",
        "title": "Background",
        "layers": [
    }


Baumlayer
~~~~~~~~~

Ein Baumlayer ist in der Lage, eine ganze Hierarchie von Layern aus einer WMS- oder QGIS-Quelle darzustellen. Ein Baumlayer wird als Gruppe im Client und mit Quelllayern als Unterknoten (oder *leaves*) angezeigt.

Es ist auch möglich, nur bestimmte Layer aus einer Quelle auszuwählen. Beim Lesen der Quelle erzeugt der Server eine virtuelle *path*-Eigenschaft für jeden Layer, welche die eindeutige ID des Layers und seine übergeordneten IDs enthält. Ähnlich den Pfaden des Dateisystems, wie ``/root-layer-id/grandparent-id/parent-id/layer-id``. Das ``pathMatch`` regex kann verwendet werden, um Ebenen mit passenden Pfaden zu filtern.


QGIS-Layer
~~~~~~~~~~

QGIS-Layer sind ähnlich wie Baumlayer zu behandeln, funktionieren jedoch nur mit QGIS-Layern. Anstelle eines einzelnen ``pathMatch`` können sie eine Liste von Matchregeln besitzen, welche dem Server sagen, wie er mit passenden QGIS-Layern umgehen soll. Sie können z. B. eine bestimmte Ebene "tilify" oder einen bestimmten Teilbaum zu einer Ebene "flatten"::


    {
        "type": "qgis",
        "title": "My qgis project",
        "path": "/data/path/to/my-project.qgis",
        "directRender": ["wms"]
    }


Vektorlayer
~~~~~~~~~~~

Vektorlayer werden auf dem GBD WebSuite Client gerendert. Wenn eine Vektorebene angefordert wird, sendet der Server die GeoJSON-Liste der Features und Stilbeschreibungen an den Client. Dieser führt dann das eigentliche Rendering durch.


WMS-Layer
~~~~~~~~~

Sie können angeben, welcher Layer verwendet werden soll. Alle WMS-Layer werden neu projiziert, kombiniert und als ein Bild angezeigt::

    {
        "type": "wms",
        "title": "Webatlas.de",
        "sourceLayers": {
            "names": ["dtk250"]
        },
        "url": "http://sg.geodatenzentrum.de/wms_dtk250"
    }


WMTS-Layer
~~~~~~~~~~

Wenn der Dienst mehrere Layer bereitstellt, können Sie angeben, welcher Layer verwendet werden soll ::


    {
        "type": "wmts",
        "title": "NRW geobasis",
        "sourceLayer": "DOP",
        "url": "https://www.wmts.nrw.de/geobasis/wmts_nw_dop"
    }


SQL-Layer
~~~~~~~~~

SQL-Layer übernehmen Geometrien aus der SQL-Tabelle. Der Datenbankanbieter und die zu verwendende Tabelle muss individuell angeben ::

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
