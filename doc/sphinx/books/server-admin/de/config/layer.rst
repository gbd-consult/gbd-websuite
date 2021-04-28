Layer
=====

Ein *Layer* in einem GBD WebSuite Projekt wird durch seinen ``type`` identifiziert. Grundsätzlich unterteilen sich Layer in  *Raster-* und *Vektorlayern*. Ein Rasterlayer stellt Geoinformation in der Pixelform dar. Die Rasterlayer enthalten keine Sachdaten (Attribute), sie können jedoch mit Suchquellen kombiniert werden, die Sachdaten enthalten. Ein Vektorlayer ist eine Sammlung von *Features*, aus geometrischen Objekten (Punkte, Linien und Polygone), zu denen auch Sachdaten zugeordnet werden. Die Darstellung von Vektorlayern können Sie mit Styling (s. ^style) frei gestalten.

Layer ``uid``
-------------

Damit die Layer in dem System eindeutig identifiziert werden können, sind die Layer-IDs (``uid``) immer mit dem Projekt- und Map-``uid`` anzugeben (die Map-``uid`` ist immer ``map``). Zum Beispiel, wenn Sie diese Konfiguration haben: ::

    {
        "projects": [
            {
                "uid": "meinprojekt",
                "map": {
                    "layers": [
                        {
                            "uid": "strassen"
                            ...

und Sie auf den Layer ``strassen`` woanders verweisen wollen (z.B. für Einrasten Funktion in ^dimension), dann ist die ``uid`` wie folgt anzugeben: ::

    {
        "type": "dimension",
        "layers": ["meinprojekt.map.strassen"]
        ...

Dies betrifft auch die URLs die das System z.B. für Map-Darstellung verwendet, z.B. ::

    http://example.com/_?cmd=mapHttpGetBox&projectUid=meinprojekt&layerUid=meinprojekt.map.strassen

Konfiguration
-------------

^REF gws.types.ext.layer.Config

Bei einer Layerkonfiguration stehen viele Optionen zur Verfügung. Die wichtigsten davon sind:

{TABLE}
``clientOptions`` | Optionen für darstellung im Client (s. ^client)
``display`` | Anzeige-Modus (``box`` - den ganzen Kartenausschnitt rendern, ``tile`` - gekachelte Darstellung, ``client`` - Darstellung im Client mit Javascript)
``legend`` | Optionen für Legenden
``meta`` | Layer-Metadaten (s. ^meta)
``opacity`` | Transparenz als eine Fließkommazahl
``ows`` |  Eigenschaften im Bezug auf OWS Dienste (s. ^ows)
``search`` | Such-Provider (s. ^search)
``templates`` | Vorlagen
``title`` | Layer-Titel
{/TABLE}

Außerdem können Sie mit ``extent``, ``extentBuffer`` und ``zoom`` die entsprechenden Eigenschaften der Karte (s. ^map) überschreiben.

Für Vektorlayer sind zusätzlich diese Optionen vorhanden:

{TABLE}
``dataModel`` | Feature Transformationsregeln (s. ^feature)
``edit`` und ``editDataModel`` | unterstützt Editieren von Layern (s. ^digitize)
``loadingStrategy`` | steuert das Laden von Features (``all`` - alle laden, ``bbox`` - nur Features im Kartenausschnitt laden)
``style`` | Style für Features (s. ^style)
{/TABLE}

Für Rasterlayer können Sie mit ``cache`` und ``grid`` das Cachen von Rasterdaten steuern (s. ^cache).

Quell-Layer
-----------

Die Layer die an externe Dienste angebunden sind (``wms``, ``wfs``, ``wmts``) sowie QGIS Layer bestehen normalerweise aus mehreren Quell-Layer. Jeder Quell-Layer, unabhängig von der Art der Quelle, hat in GWS folgende Eigenschaften:

{TABLE}
``name`` | Name, entspricht ``<Layer><Name>`` in WMS und ``<FeatureType><Name>`` in WFS
``uid`` | Die ID wird aus dem Namen generiert, z.B. aus dem Namen "Bäume" entsteht die ID "baeume"
``path`` | Layer Pfad, die ID von Überlayer, mit einem Punkt getrennt
``level`` | Die Tiefe des Layers, wobei der Root-Layer Tiefe ``1`` hat
{/TABLE}

Zum Beispiel, wenn ein WMS Dienst folgende Struktur hat: ::

    <Layer>
        <Name>Natur</Name>
        ...
        <Layer>
            <Name>Forst</Name>
            ...
            <Layer>
                <Name>Bäume</Name>

entstehen in GWS folgende Quell-Layer:

{TABLE head}
``name`` | ``uid`` | ``path`` | ``level``
``Natur`` | ``natur`` | ``natur`` | 1
``Forst`` | ``forst`` | ``natur.forst`` | 2
``Bäume`` | ``baeume`` | ``natur.forst.baeume`` | 3
{/TABLE}

Wenn Sie nur bestimmte Quell-Layer in Ihrem Projekt zeigen wollen, können Sie mit der Option ``sourceLayers`` die Quell-Layers nach Namen (``names``), Pfaden (``pattern``) oder Tiefe (``level``) filtern.

Vorlagen
--------

Für einen Layer können Sie eine Vorlage mit dem ``subject`` ``layer.description`` definieren, die im Client gezeigt wird, wenn der Nutzer den Layer auswählt. Bei Vektorlayern können zusätzlich Feature-Vorlagen definiert werden (s. ^feature).

Legenden
--------

^REF gws.common.layer.types.LegendConfig

Mit der Option ``legend`` können Sie die Legende für den Layer konfigurieren. Sie haben die Wahl zwischen:

- keine Legende für den Layer zeigen. In dem Fall setzen Sie ``enabled`` auf ``false``
- automatische Legende zeigen (z.B. für WMS Layer): setzen Sie ``enabled`` auf ``true``
- ein Bild als Legende zeigen, geben Sie die ``url`` des Bildes an
- eine HTML Vorlage verwenden für eine komplexe Legende. Nutzen Sie die ``template`` Eigenschaft

Layer Typen
-----------

geojson
~~~~~~~

^REF gws.ext.layer.geojson.Config

Vektorlayer, der die Daten aus einer GeoJSON (https://geojson.org/) Datei darstellt.

group
~~~~~

^REF gws.ext.layer.group.Config

Gruppenlayer enthalten andere Layer, sie liefern selbst keine Geodaten. Neben der visuellen Gruppierung besteht ein weiterer Zweck einer Gruppe darin, die Zugriffs- bzw Ausmaß-Konfigurationen für ihre untergeordneten Layer beizubehalten. Eine Gruppe kann "virtuell" oder ``unfolded`` erstellt werden, in diesem Fall wird sie im Client nicht angezeigt, während ihre untergeordneten Layer vorhanden sind.

postgres
~~~~~~~~

^REF gws.ext.layer.postgres.Config

Vektor-Layer, übernimmt Geometrien aus einer Postgres-Tabelle. Sie müssen die zu verwendende Tabelle angeben ::

        {
            "title": "Houses",
            "type": "postgres",
            "table": {
                "name": "myschema.mytable"
            },
            "loadingStrategy": "bbox",
            "style": {
                "type": "css",
                "text": "stroke:rgb(0,255,0); stroke-width:1px; fill:rgba(0,100,0,0.2)"
            }
        }

Mehr über Tabellen-Konfigurationen siehe in ^db.

qgis/qgisflat
~~~~~~~~~~~~~

QGIS Layer, mehr dazu in ^qgis.

tile
~~~~

^REF gws.ext.layer.tile.Config

Rasterlayer, zeigt Kachel aus einer XYZ-Kachelquelle (wie z.B. OSM). Sie müssen die URL mit Platzhaltern ``{x}``, ``{y}`` und ``{z}`` angeben ::

    {
        "type": "title",
        "title": "Open Street Map",
        "url": "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
    }

wfs
~~~

^REF gws.ext.layer.wfs.Config

Vektorlayer, zeigt Features aus einen WFS Dienst ::

        {
            "type": "wfs",
            "title": "Geobasis NRW WFS Service",
            "url": "https://www.wfs.nrw.de/geobasis/wfs_nw_dvg",
            "sourceLayers": {
                "pattern": "nw_dvg1_gem"
            }
        }

wms
~~~

^REF gws.ext.layer.wms.Config

Rasterlayer, zeigt Rasterdaten aus einem WMS Dienst. Falls der Dienst mehrere Layer enthält, werden diese als eine Baumstruktur dargestellt ::

    {
        "type": "wms",
        "title": "Webatlas.de - Alle Layer",
        "url": "http://sg.geodatenzentrum.de/wms_dtk250"
    }

wmsflat
~~~~~~~

^REF gws.ext.layer.wmsflat.Config

Rasterlayer, zeigt Rasterdaten aus einem WMS Dienst. Die WMS-Layer werden kombiniert, ggf. umprojiziert,  und als einzelnes Bild angezeigt ::

    {
        "type": "wmsflat",
        "title": "Webatlas.de - DTK250",
        "sourceLayers": {
            "names": ["dtk250"]
        },
        "url": "http://sg.geodatenzentrum.de/wms_dtk250"
    }

wmts
~~~~

^REF gws.ext.layer.wmts.Config

Rasterlayer, zeigt Rasterdaten aus einem WMTS Dienst ::

    {
        "type": "wmts",
        "title": "Geobasis NRW WMTS Service",
        "sourceLayer": "DOP",
        "url": "https://www.wmts.nrw.de/geobasis/wmts_nw_dop"
    }
