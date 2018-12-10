Basis Konzept
==============

Anforderungen und URL-Adressen
------------------------------


Einmal gestartet, hört die GBD WebSuite auf Ports ``80/443`` und verarbeitet ``GET`` und ``POST`` Anfragen. Wie ein herkömmlicher Webserver kann GWS statische Inhalte wie HTML-Seiten oder Bilder bereitstellen, aber sein Hauptzweck ist es, dynamische Kartenbilder und Daten bereitzustellen. Für dynamische Anfragen gibt es einen einzigen Endpunkt (url), nämlich den ``_`` (Unterstrich). Alle Anfragen an diesen Endpunkt müssen den Befehl (``cmd``) Parameter enthalten.
Zusätzlich müssen alle ``POST`` Anfragen im JSON-Format vorliegen

Hier ein paar Beispiele von Anfragen, die GBD WebSuite bearbeiten kann:: 

 ## normale Webanfrage 

    http://maps.my-server.com/images/smile.jpg

    ## dynamische GET-Anfrage (z. B. Kartenbilder) 

    http://maps.my-server.com/_?cmd=mapHttpGetBbox&layer=london.metro&width=100&height=200&bbox=10,20,30,40

    ## dynamische POST-Anfrage (z.B. Suche)

    {
        "cmd":"searchRun",
        "params": {
            "projectUid": "london",
            "bbox": [10,20,30,40],
            "keyword": "queen"
        }
    }

Dynamische GET-URLs können durch URL-Rewriting modifiziert werden, so dass diese :: 

    http://maps.my-server.com/wms/london/metro

kann anstelle verwendet werden::

    http://maps.my-server.com/_?cmd=wmsHttpGetMap&project=london&layers=metro

Standorte und Projekte
-----------------------

Auf der obersten Ebene arbeitet die GBD WebSuite mit zwei Arten von Entitäten: *Projekte* und *Standorte*. Ein Projekt ist in etwa eine Karte und eine Sammlung von Einstellungen, die sich auf diese spezielle Karte beziehen. Eine Site ist ein Domain-Name, der an Autorisierungs- und Routing-Regeln gebunden ist.

Im obigen Beispiel ist ``london`` ein Projekt, ``metro`` ist eine für dieses Projekt konfigurierte Ebene, während der Domainname ``maps. my-server. com`` und die entsprechende Rewrite-Regel aus der Site-Konfiguration übernommen werden.

Sites und Projekte sind orthogonale Konzepte, und Sie können dasselbe Projekt unter mehreren Sites ausführen. Wenn Sie z. B. ``maps.my-server.com`` to e.g. ``gis.my-other-server.com`` ändern würden, würde dies keine Änderungen im Projekt ``london`` erfordern.

Aktionen
-----------

Der Befehlssatz (``cmd`` in den obigen Beispielen) ist nicht festgelegt und kann frei konfiguriert werden. Befehle sind in *Aktionen* gruppiert, Sie können verfügbare Aktionen global oder projektbezogen konfigurieren. 


Karten, Ebenen und Quellen
---------------------------------

Jedes GBD WebSuite Projekt enthält mindestens eine *Map*, die eine Sammlung von *Layern* ist. Es gibt verschiedene Arten von Ebenen (z. B. "Box" oder "Kachel"). Sie können Zugriffsrechte, Ansichtseigenschaften (wie ein Extent) und die Metadaten für die gesamte Karte und für jede Ebene individuell konfigurieren. Die meisten Ebenen sind auch an *source* Objekte gebunden, die dem Server mitteilen, woher die Geodaten stammen. Eine Layer-Konfiguration enthält typischerweise Anweisungen für den Server, wie die Quelldaten transformiert werden, z. B.

- die Daten transformieren
- die Bilder von WMS in Kacheln umwandeln und umgekehrt
- Merkmaldaten neu formatieren
- benutzerdefinierte Stile auf Features anwenden


Pluggable Architektur
----------------------

Fast alle Funktionen der GBD WebSuite sind über Plugins implementiert. Wir haben Plugins für diese Art von Objekten

TABLE
   Aktionen ~ Server-Aktionen
   Autorisierungsanbieter ~ Autorisierung und Authentifizierung handhaben
   Datenbankanbieter ~ Datenbankverbindungen
   Suche bietet ~ Volltextsuche und Attributsuche
   Ebenen ~ Kartenebenen
   Quellen ~ Geodatenquellen für Karten
   Druckvorlagen ~ Verschiedene Druckvorlagenformate
/TABLE

Pluggable-Objekte in der Konfiguration werden durch ihre ``type`` Eigenschaft identifiziert


Konfigurationsdateien und Objekte
-----------------------------------

GWS unterstützt verschiedene Konfigurationsformate:

- json, in diesem Fall muss der Name der Konfigurationsdatei mit ``config. json`` enden.
- yaml (``config. yaml``). Wir verwenden Json in diesen Dokumenten, aber Sie können Yaml immer mit der gleichen Struktur verwenden, wenn Sie es mehr wollen.
- python (``config. py``). Komplexe, sich wiederholende oder hochdynamische Konfigurationen können auch in gerader Pythonform geschrieben werden. Ihr Python-Skript muss eine Funktion namens ``config()`` mit der gleichen Struktur wie JSON enthalten. Beachten Sie, dass Ihr Konfigurationsmodul innerhalb des Containers ausgeführt wird und daher mit Python 3. 6 kompatibel sein muss.

Die Konfiguration beginnt mit der Hauptkonfigurationsdatei (``GWS_CONFIG``), die weitere Konfigurationsdateien für Projekte und Projektgruppen enthalten kann. Sobald alle Dateien gelesen und gepaart sind, werden alle konfigurierten Objekte zu einem großen "Baum" zusammengefasst, wobei das ``Application`` Objekt der Wurzelknoten ist. Hier ist ein Beispiel für einen solchen Baum::

   Application
    |
    |-- auth options
    |-- server options
    |-- web options
    |
    \-- projects
        |
        |-- First project
        |   |-- project options
        |   \-- Map
        |       |-- First layer
        |       \-- Second layer
        |
        \-- Second project
           |-- project options
           \-- Map
               \-- Layer group
                   \-- Sub-layer


Die meisten Konfigurationsoptionen sind vererbbar, d. h. wenn das System nach einer Eigenschaft für eine Ebene sucht und diese nicht explizit konfiguriert ist, dann wird die übergeordnete Ebene, dann die Karte, dann das enthaltene Projekt und schließlich die Wurzel ``Application`` konsultiert. 
