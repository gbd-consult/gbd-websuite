Web Server
==========

Web-Inhalte (HTML-Dateien, Bilder, Downloads) werden in der GBD WebSuite von dem integrierten NGINX, einem schnellen und zuverlässigen Web-Server, verarbeitet.

Sie können mehrere *Sites* ("virtuelle Hosts") in einer einzigen GBD WebSuite Installation konfigurieren, jeder mit seinem eigenen Hostnamen und Dokumentenstamm.


Statische Dateien und Assets
------------------------------

Es gibt zwei Arten von Webinhalten: *statische* Ressourcen und *Vermögenswerte*. Statische Ressourcen werden in einem Web-Root-Ordner abgelegt und ohne jegliche Bearbeitung bereitgestellt. Assets sind Ressourcen, die serverseitig verarbeitet werden müssen, bevor sie dem GBD WebSuite Client zur Verfügung gestellt werden können.

Anwendungsfälle für statische Dateien sind:

- einfache, öffentliche html-Seiten
- Javascript- und CSS-Dateien (einschließlich des GBD WebSuite Client Materials)
- statische Bilder

Anwendungsfälle für Vermögenswerte sind:

- jede Ressource, die einer Autorisierung bedarf
- Template-basierte HTML-Seiten (Die GBD WebSuite verwendet `mako <https:--www.makotemplates.org/>`_ für das Templating)
- projektspezifische Ressourcen

Die GBD WebSuite bedient nur Ressourcen mit bekannten Mime-Typen (bestimmt durch die Dateiendung), die Voreinstellung ist ::

    .css .csv .gif .html .jpeg .jpg .js .json .pdf .png .svg .ttf .xml .zip

Sie können diese Liste pro Seite oder pro Projekt neu definieren

Rewrite-Regeln
----------------

Assets werden vom Serverbefehl ``assetHttpGetPath`` behandelt, der den Parameter ``path`` akzeptiert, und optional eine eindeutige Projekt-ID, so dass die endgültige URL wie folgt lautet::

    http://example.org/_?cmd=assetHttpGetPath&path=somepage.mako

Die folgende Rewrite-Regel :: 

    {
        "match": "^/([a-z]+)/([a-z]+)$",
        "target": "_?cmd=assetHttpGetPath&projectUid=$1&path=$2.mako"
    }


wird diese URL in einfach umwandeln::

    http://example.org/myproject/somepage

Das ``match`` ist ein erforderlicher Ausdruck und das ``target`` ist die endgültige URL mit ``{$n}`` Platzhaltern, die den Capture-Gruppen im RegEx entsprechen. Wenn das Ziel mit einem Schema beginnt (z. B. ``http://``), führt der Server einen Redirect statt eines Rewritings durch.

Website-Konfiguration
-----------------------------

Eine Website-Konfiguration muss einen Hostnamen (Hostname ``*`` markiert die Standard-Site), eine Root- und Asset-Konfiguration sowie optional eine Reihe von URL-Rewriting-Regeln enthalten ::


    {

        ## Hostname


        "host": "example.org",

        ## statisches document root

        "root": {

            ## absoluter Pfad zur root directory

            "dir": "/example/www-root",

            ## erlaubte Dateierweiterungen (zusätzlich zur Standardliste) 

            "allowMime": [".xls", ".doc"],

            ## deaktivierte Dateierweiterungen (aus der Standardliste) 

            "denyMime": [".xml", ".json"],

        },

        ## assets root

        "assets": {

            ## absoluter Pfad zum Site-Asset-Verzeichnis

            "dir": "/example/www-assets",

        },

        ## rewrite rules

        "rewrite": [

            {
                "match": "^/$",
                "target": "_?cmd=assetHttpGetPath&path=root-page.mako"
            },
            {
                "match": "^/hello/([a-z]+)$",
                "target": "_?cmd=assetHttpGetPath&projectUid=hello_project&path=$1.mako"
            }
        ]


Projektressourcen
----------------------

Jedes GBD WebSuite Projekt kann seine eigene Asset-Root-Konfiguration haben. Wenn der Client ein Asset ohne Projekt-UID anfordert, z. B. ::

    http://example.org/_?cmd=assetHttpGetPath&path=somepage.mako

dann wird das Asset im Site-Asset-Verzeichnis gesucht. Wenn ein Auftrag mit einem Projekt uid ::

    http://example.org/_?cmd=assetHttpGetPath&projectUid=myproject&path=somepage.mako

dann wird das Asset zuerst in den Projekt-Assets gesucht, wenn es nicht gefunden wird, wird das Site-Asset-Verzeichnis als Fallback verwendet.
