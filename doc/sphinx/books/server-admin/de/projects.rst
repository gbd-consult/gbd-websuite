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
-----------------------

Eine Projektkonfiguration sollte mindestens die Konfiguration ``title`` und eine ``Map`` enthalten. Einige Optionen, wie ``access``, ``assets`` und ``client`` überschreiben die jeweiligen Anwendungs- und Webseitenoptionen. ``printer`` beschreibt Projektdruckvorlagen (siehe `doc:`print`) ::

    {
        "title": "Hello",
        "uid": "project1",
        "map": {
          "view": {
              "extent": [554000, 6461000, 954000, 6861000],
              "scales": [1e3, 5e3, 1e4]
          },
          "layers": [
                {
                  "title": "OpenStreetMap",
                  "type": "client",
                  "kind": "osm"
                }
            ]
        }
    }


Karten
------

Eine *Karte* ist im Grunde eine Sammlung von *Ebenen* (siehe `doc:`Ebenen`). Eine Kartenkonfiguration kann auch Cache-, Raster- und Ansichtsoptionen enthalten, die als Fallback für Ebenen dienen, die diese nicht explizit definieren. Die Option ``crs`` muss ein gültiger EPSG CRS-Referenzstring sein. Alle Ebenen im Projekt werden im CRS angezeigt. Quellen mit unterschiedlichen Projektionen werden dynamisch wiedergegeben.


Multi-Projekte
--------------

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


Neben regulären Platzhaltern werden folgende Platzhalter unterstützt: ::

        TABLE
        `` {{path}} `` ~ vollständiger Pfad der aktuellen Datei
        `` {{dirname}} `` ~ Verzeichnisname der aktuellen Datei
        `` {{filename}} `` ~ Dateiname der aktuellen Datei
        `` {{index}} `` ~ Index der aktuellen Datei in der Liste
        /TABLE


Projekt HTML Seite
------------------

Um Ihr Projekt in einem Webbrowser anzuzeigen, benötigen Sie eine HTML-Seite, die unseren Javascript-Client (siehe: doc: `client`) und die Projekt-ID enthalten sollte, damit der Client weiß, welches Projekt geladen werden soll. Auf der Seite muss sich ein div-Element mit dem Klassennamen gws befinden. Hier wird die Client-Benutzeroberfläche geladen. Ansonsten können Sie Ihre Startseite frei gestalten. Hier ist ein Beispiel ::

    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8"/>
        <title>My First Project!</title>

        <!-- Load the Client, 2.3.1 is the version you're using -->
        <link rel="stylesheet" href="/gws-client/gws-light-2.3.1.css" type="text/css">
        <script src="/gws-client/gws-vendor-2.3.1.js"></script>
        <script src="/gws-client/gws-client-2.3.1.js"></script>

        <!-- Position the Client as you wish -->
        <style>
            .gws {
                position: fixed;
                left: 10px;
                top: 20px;
                right: 40px;
                bottom: 50px;
            }
        </style>

        <!-- Project uid, as defined in the config file -->
        <script>
            GWS_PROJECT_UID = "project1";
        </script>

        <!-- Your own css, if needed -->
        <link rel="stylesheet" href="/my-style.css" type="text/css">

        <!-- Your additional css/scripts and other resources -->

    </head>

    <body>
        <!-- This is where the Client will be loaded -->
        <div class="gws"></div>

        You can add more content here...
    </body>
    </html>

Platzieren Sie diese Datei in Ihrem konfigurierten ``web``-Verzeichnis (siehe: doc: "web"), um sie im Web zur Verfügung zu stellen.
