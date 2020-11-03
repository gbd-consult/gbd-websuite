Drucken
=======

Ein Projekt in der GBD WebSuite kann eine oder mehrere Druckvorlagen bereitstellen. Bei mehreren Vorlagen wird dem Nutzer im Client ein Menü angezeigt, wo die Vorlage und ggf. die Qualitätsstufe gewählt werden kann.

Qualitätsstufen
---------------

Für jede Vorlage können Sie eine Liste der Qualitätsstufen konfigurieren. Eine Qualitätsstufe ist ein DPI-Wert mit einem Namen. Beachten Sie, dass das Drucken mit hohen DPI-Werten viel Speicherplatz benötigt und nicht einmal mit Quellen möglich ist, die Beschränkungen für die Begrenzung von Anfragen auferlegen. Das Drucken einer A3-Karte mit 300 DPI wird nicht immer funktionieren.

Zwei DPI-Werte werden in GWS auf besondere Weise bearbeitet:

- bei dem DPI Wert ``0`` wird  die Karte als Bitmap-Bild (*Screenshot*) gedruckt. Diese Methode bietet niedrige Qualität an, ist jedoch sehr schnell.
- bei jedem DPI Wert größer als ``0`` und kleiner als ``90`` werden die Rasterlayer als Screenshots gedruckt, die Vektoren aber in der voller Auflösung. Verwenden Sie diese Methode wenn die Vektoren (wie z.B. Markierungen) wichtig sind, die Hintergrundkarte aber nur zur Orientierung dient.

Druckvorlagen
-------------

html
~~~~

Eine ``html`` Druckvorlage ist eine Vorlage (s. ^template) die auch spezielle HTML-Tags enthält, der beim Drucken mit aktuellen Werten ersetzt werden. Es werden folgende Tags unterstützt:

{TABLE head}
Tag | Bedeutung | Beispiel
``gws:page`` | Seiteneigenschaften | ``<gws:page width="297" height="210" margin="5 5 5 5"/>``
``gws:map`` | Karte einfügen | ``<gws:map width="150" height="150"/>``
``gws:legend`` | Legende einfügen | ``<gws:legend layer="london.map.metro"/>``
``gws:header`` | Kopfzeile | ``<gws:header>Meine Karte</gws:header>``
``gws:footer`` | Fußzeile | ``<gws:footer>Seite {page} von {page_count}</gws:footer>``
{/TABLE}

Zusätzlich zu allgemeinen Variablen, können Sie in Druckvorlagen auch folgende verwenden:

{TABLE}
``page`` | Seitennumer
``page_count`` | Anzahl der Seiten
{/TABLE}

Eine ``html`` Druckvorlage kann z.B. wie folgt aussehen: ::

    <h1>Meine Karte</h1>
    <gws:map width="150" height="150"/>
    <p>Erstellt am {date.short} vom Nutzer {user.displayName}</p>

qgis
~~~~

Die QGIS Druckvorlagen sind unter ^qgis beschrieben.

Attribute
---------

Wenn Sie von Nutzer editierbare Attribute auf dem Ausdruck benötigen, können Sie diese mit einem Datenmodell (``dataModel``) in der Vorlagenkonfiguration definieren. Die Attribute dieses Models sind vom Nutzer editierbar und sind in einer ``html`` Vorlage als Variablen verfügbar. Zum Beispiel, wenn Sie eine Druckvorlage wie folgt definieren: ::

    {
        "type": "html",
        ...
        "dataModel": {
            "rules": [
                {
                    "name": "title",
                    "title": "Überschrift",
                    "type": "str"
                },
                {
                    "name": "place",
                    "title": "Ort",
                    "type": "str"
                }
            }
        ]
    }

wird dem Nutzer beim Drucken ein Formular mit den Feldern "Überschrift" und "Ort" gezeigt, wobei Sie in Ihrer Vorlage die Variablen ``{title}`` und ``{ort}`` an beliebigen Stellen nutzen können. Für QGIS Vorlage können Sie auch QGIS Syntax ``[% @title %]`` nutzen.

Für mehr Info on Datenmodelle s. ^feature.
