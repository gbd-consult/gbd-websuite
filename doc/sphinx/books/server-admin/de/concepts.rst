Basis Konzept
==============

Anforderungen und URL-Adressen
------------------------------


Einmal gestartet, greift die GBD WebSuite auf die Ports ``80/443`` zu und verarbeitet ``GET`` und ``POST`` Anfragen. Wie ein herkömmlicher Webserver kann die GBD WebSuite statische Inhalte wie HTML-Seiten oder Bilder bereitstellen. Ihr Hauptzweck ist jedoch die Bereitstellung von dynamischen Kartenbildern und Daten. Für dynamische Anfragen (url) gibt es einen einzigen Endpunkt, nämlich den ``_`` (Unterstrich). Alle Anfragen an diesen Endpunkt müssen den Befehl (``cmd``) Parameter enthalten.
Zusätzlich müssen alle ``POST`` Anfragen im JSON-Format vorliegen.

Hier sind ein paar Beispiele von Anfragen, die die GBD WebSuite bearbeiten kann::

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

Dynamische GET-URLs können durch URL-Rewriting modifiziert werden, so dass::

    http://maps.my-server.com/wms/london/metro

anstelle von::

    http://maps.my-server.com/_?cmd=wmsHttpGetMap&project=london&layers=metro

verwendet werden kann.


Standorte und Projekte
-----------------------

Auf der obersten Ebene arbeitet die GBD WebSuite mit zwei Arten von Strukturen. Zum einen mit *Projekten* und zum anderen mit *Standorten*. Ein *Projekt* ist in etwa eine Karte mit einer Sammlung von Einstellungen, die sich auf diese spezielle Karte beziehen. Ein *Standort* ist ein Domain-Name, der an Autorisierungs- und Routing-Regeln gebunden ist.

Im obigen Beispiel ist ``london`` ein Projekt, ``metro`` ist eine für dieses Projekt konfigurierte Ebene, während der Domainname ``maps. my-server. com`` und die entsprechende Rewrite-Regel aus der Site-Konfiguration übernommen werden.

Standorte und Projekte sind orthogonale Konzepte. Sie können dasselbe Projekt unter mehreren Standorten ausführen. Wenn Sie zum Beispiel ``maps.my-server.com`` zu ``gis.my-other-server.com`` ändern würden, würde dies keine Änderungen im Projekt ``london`` erfordern.

Aktionen
-----------

Die Liste der Befehle (``cmd`` in den obigen Beispielen) ist nicht festgelegt und kann frei konfiguriert werden. Befehle sind in *Aktionen* gruppiert, welche Sie können verfügbare Aktionen global oder projektbezogen konfigurieren.


Karten, Ebenen und Quellen
---------------------------------

Jedes GBD WebSuite Projekt enthält mindestens eine *Map*, die eine Sammlung von *Layern* ist. Es gibt verschiedene Arten von Ebenen (z. B. "Box" oder "Kachel"). Sie können Zugriffsrechte, Ansichtseigenschaften (wie ein Extent) und die Metadaten für die gesamte Karte und für jede Ebene individuell konfigurieren. Die meisten Ebenen sind an *source* Objekte gebunden, die dem Server mitteilen, woher die Geodaten stammen. Eine Layer-Konfiguration enthält typischerweise Anweisungen für den Server, wie die Quelldaten transformiert werden. Zum Beispeiel

- die Daten transformieren
- die Bilder von WMS in Kacheln umwandeln und umgekehrt
- Merkmaldaten neu formatieren
- benutzerdefinierte Stile auf Features anwenden


Steckbare Architektur
----------------------

Fast alle Funktionen der GBD WebSuite sind über Plugins implementiert. Zu folgenden Arten von Objekten sind Plugins vorhanden:

TABLE
   Aktionen ~ Server-Aktionen
   Autorisierungsanbieter ~ Autorisierung und Authentifizierung handhaben
   Datenbankanbieter ~ Datenbankverbindungen
   Suche bietet ~ Volltextsuche und Attributsuche
   Ebenen ~ Kartenebenen
   Quellen ~ Geodatenquellen für Karten
   Druckvorlagen ~ Verschiedene Druckvorlagenformate
/TABLE

Steckbare Objekte werden in der Konfiguration durch ihre ``type`` Eigenschaft identifiziert.


Konfigurationsdateien und Objekte
-----------------------------------

Die GBD WebSuite unterstützt verschiedene Konfigurationsformate:

- JSON, in diesem Fall muss der Name der Konfigurationsdatei mit ``config. json`` enden.
- YAML (``config. yaml``). Wir verwenden JSON in diesen Dokumenten, aber Sie können auch YAML mit der gleichen Struktur verwenden.
- Python (``config. py``). Komplexe, sich wiederholende oder hochdynamische Konfigurationen können auch in Pythonform geschrieben werden. Ihr Python-Skript muss eine Funktion namens ``config()`` mit der gleichen Struktur wie JSON enthalten. Beachten Sie, dass Ihr Konfigurationsmodul innerhalb des Containers ausgeführt wird und daher mit Python 3.6 kompatibel sein muss.

Die Konfiguration beginnt mit der Hauptkonfigurationsdatei (``GWS_CONFIG``), welche weitere Konfigurationsdateien für Projekte und Projektgruppen enthalten kann. Sobald alle Dateien gelesen und gepaart sind, werden alle konfigurierten Objekte zu einem großen "Baum" zusammengefasst, wobei das ``Application`` Objekt der Wurzelknoten ist. Hier ist ein Beispiel für einen solchen Baum::

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
