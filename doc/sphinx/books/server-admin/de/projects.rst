Projekte
========

Ein *Projekt* in der GBD WebSuite besteht aus einer Karte, Druckvorlagen und zusätzlichen Optionen. Außerdem können Sie einige Anwendungsoptionen für jedes Projekt einzeln überschreiben.

Projektstandorte
-----------------

Es gibt drei Möglichkeiten, Projekte zu Ihrer GBD WebSuite Installation hinzuzufügen (Sie können sie auch kombinieren):

TABLE
``projects`` ~ Projekte direkt in der Hauptkonfiguration konfigurieren
``projectPaths`` ~ enthält spezifische Projektkonfigurationen
``projectDirs`` ~ enthält alle Projekte aus Verzeichnissen
/TABLE

Bei Verwendung von ``projectDirs`` durchsucht der Server die angegebenen Verzeichnisse rekursiv und fügt alle Dateien hinzu, die mit ``config. py`` oder ``config. json`` oder ``config. yaml`` enden.

Jede Projektkonfigurationsdatei kann eine Konfiguration für ein einzelnes Projekt oder ein Array solcher Konfigurationen enthalten.

Projekt-Konfigurationen
------------------------------

Eine Projektkonfiguration sollte mindestens die Konfiguration ``title`` und eine ``Map`` enthalten. Einige Optionen, wie ``access``, ``assets`` und ``client`` überschreiben die jeweiligen Anwendungs- und Webseitenoptionen. ``printer`` beschreibt Projektdruckvorlagen (siehe `doc:`print`).


Karten
------------

Eine *Karte* ist im Grunde eine Sammlung von *Ebenen* (siehe `doc:`Ebenen`). Eine Kartenkonfiguration kann auch Cache-, Raster- und Ansichtsoptionen enthalten, die als Fallback für Ebenen dienen, die diese nicht explizit definieren. Die Option ``crs`` muss ein gültiger EPSG CRS-Referenzstring sein. Alle Ebenen im Projekt werden im CRS angezeigt. Quellen mit unterschiedlichen Projektionen werden dynamisch wiedergegeben.

Multi-Projekte
----------------------

Eine Projektkonfiguration kann auch als Vorlage für mehrere Projekte dienen. Um ein Template einzurichten, setzen Sie ``multi`` auf ``true`` und geben Sie einen regulären Ausdruck in ``multiMatch`` ein. Der Server zählt alle Dateien auf dem Server auf, die diesem Ausdruck entsprechen, und erstellt eine Projektkonfiguration für jede Datei, indem er regex-Platzhalter ``{$n}`` in anderen Optionen Werte ersetzt. Zum Beispiel wird diese Vorlage alle QGIS-Karten in ``/data/qgis-maps`` auflisten und ein Projekt mit einer QGIS-Ebene für jede gefundene Karte erstellen:: 

        "multi": True,
        "multiMatch": "/data/qgis-maps/(.+?).qgs$",
        "title": "Project {$1}",
        "map": {
            "layers": [
                {
                    "type": "qgis",
                    "title": "{$1}",
                    "source": {
                        "type": "qgis",
                        "map": "{$0}"
                    },
                },
            ]
        }









