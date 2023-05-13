Grundkonzepte
=============

In diesem Kapitel beschreiben wir kurz, auf welchen Grundkonzepten die Funktionalitäten der GBD WebSuite basieren.

Anfragen und URLs
-----------------

Einmal gestartet, hört die GBD WebSuite auf Ports ``80/443`` und verarbeitet ``GET`` und ``POST`` Anfragen. Wie ein herkömmlicher Webserver kann die GBD WebSuite statische Inhalte wie HTML-Seiten oder Bilder bereitstellen, aber der Hauptzweck liegt darin, dynamische Kartenbilder und Daten bereitzustellen. Für dynamische Anfragen gibt es einen einzigen Endpunkt (url), nämlich den ``_`` (Unterstrich). Alle Anfragen an diesen Endpunkt müssen den Befehl (``cmd``) Parameter enthalten. Zusätzlich müssen alle ``POST`` Anfragen im JSON-Format vorliegen.

Hier ein paar Beispiele von Anfragen, die die GBD WebSuite bearbeiten kann.

Eine statische GET-Anfrage: ::

    http://example.com/images/smile.jpg

Eine dynamische GET-Anfrage (z. B. ein Kartenbild): ::

    http://example.com/_?cmd=mapHttpGetBox&projectUid=london&layerUid=london.map.metro&width=100&height=200

Bei dynamischen GET-Anfragen unterstützt die GBD WebSuite eine alternative Schreibweise mit der die Parameter und Werte mit einem Slash ``/`` getrennt werden: ::

    http://example.com/_/mapGetBox/projectUid/london/layerUid/london.map.metro/width/100/height/200

Eine dynamische POST-Anfrage (z.B. Suche): ::

    {
        "cmd":"search",
        "params": {
            "projectUid": "london",
            "bbox": [10,20,30,40],
            "keyword": "queen"
        }
    }

Aktionen
--------

Anhand vom ``cmd`` Parameter entscheidet der Server welche *Server Aktion* die Bearbeitung der Anfrage übernimmt. Falls die Aktion existiert und richtig konfiguriert ist,  wird die Anfrage zu dieser Aktion weitergeleitet. Die Aktion bearbeitet die Anfrage und stellt eine Antwort bereit, die abhängig von der Natur der Anfrage, in HTML, JSON oder PNG Format vorliegt. Intern sind die Aktionen die Python-Klassen, die für jeden Befehl (``cmd``) über eine Methode verfügen. Im obigen Beispiel (``cmd=mapHttpGetBox``) ist ``map`` die Aktion und ``httpGetBox`` die Methode, die diese Anfrage bearbeitet.

%info
 Server Aktionen sind unter ^config/action beschrieben.
%end

Webseiten und Projekte
----------------------

Auf der obersten Ebene arbeitet die GBD WebSuite mit zwei Arten von Entitäten: *Projekte* und *Webseiten*. Ein Projekt ist in etwa eine Karte und eine Sammlung von Einstellungen, die sich auf diese spezielle Karte beziehen. Eine Site ist ein Domain-Name, der an Autorisierungs- und Routing-Regeln gebunden ist.

In dieser URL ::

    http://example.com/_?cmd=mapHttpGetBox&projectUid=london&layerUid=london.map.metro

ist ``london`` ein Projekt, ``london.map.metro`` ist eine für dieses Projekt konfigurierte Ebene, während der Domainname ``example.com`` aus der Webseiten-Konfiguration übernommen wird.

Webseiten und Projekte sind orthogonale Konzepte. Sie können dasselbe Projekt unter mehreren Webseiten ausführen. Wenn Sie z. B. ``example.com`` ins ``other-example.com`` ändern würden, würde dies keine Änderungen im Projekt ``london`` erfordern.

Client
------

Obwohl die GBD WebSuite als gewöhnlicher Webserver arbeiten kann, ist ihr Hauptzweck, zusammen mit einem "reichen" Javascript-Client verwendet zu werden, der in der Lage ist, dynamische Web-Maps wie OpenLayers of Leaflet anzuzeigen. Wir bieten einen solchen Client als Teil der GBD WebSuite an und stellen einige Optionen in der Serverkonfiguration zur Verfügung, um unseren Client gezielt zu unterstützen.

%info
 Mehr dazu in ^config/client.
%end

Statische Dokumente und Assets
------------------------------

*Statische Web-Dokumente* sind Dateien (z.B. HTML oder PDF) die keine spezielle Bearbeitung auf dem Server erfordern und jedem Nutzer unverändert zur Verfügung stehen. Bei einer Webseite kann ein "public" Ordner mit statischen Dokumenten konfiguriert werden, wobei die URLs den Dateipfaden entsprechen. Zum Beispiel, wenn Ihr "public"-Ordner als ``data/web`` konfiguriert wird, und Sie eine PDF Datei unter ``data/web/documents/file.pdf`` abspeichern, kann diese Datei unter ``http://example.com/documents/file.pdf`` heruntergeladen werden.

Ein *Asset* ist dagegen ein Dokument, das dynamisch erzeugt wird, abhängig vom Kontext (eine *Vorlage*) oder nur berechtigten Nutzern zur Verfügung steht. Assets werden in einem speziellen Ordner platziert, der sowohl für eine Webseite als auch Projekt-abhängig konfiguriert werden kann.

%info
 Mehr dazu in ^config/web, ^config/template und ^config/project.
%end

Autorisierung
-------------

GWS enthält eine Rollen-Basierte Autorisierung. Bei allen konfigurierbaren Systemobjekte kann mittels *Zugriffsblöcken* (``access``) spezifiziert werden, welche Rollen den Zugriff zu diesem Objekt haben. Falls es für die gegebene Rolle keine explizite Anweisung gibt, wird das übergeordnete Objekte gecheckt. Für das Root Objekt (`application`) werden per default alle Zugriffe verweigert.

Individuelle Zugangsdaten (Nutzername, Passwort) werden zu Rollen mittels *Autorisierungsanbieter* (``provider``) verknüpft. Die Aufgabe eines Anbieters ist, die Zugangsdaten gegen der angegebenen Datenquelle zu prüfen. Aktuell werden folgende Anbieter unterstützt:

* LDAP/ActiveDirectory
* file-basiert

%info
 In der Zukunft sind auch Datenbank Provider geplant.
%end

*Autorisierungsmethoden* (``method``) geben an, wie die Zugangsdaten dem System übergeben werden. Aktuell sind diese Methoden unterstützt:

- ``web``: Übergabe mittels eines Web-Formulars (Login-Form)
- ``basic``: Übergabe mittels einer HTTP-Basic Autorisierung

%info
 In der Zukunft sind auch OAuth, Two-Factor sowie Windows single sign-on (SSO) geplant.
%end

%info
 Mehr dazu in ^config/auth.
%end

Karten und Layer
----------------

Jedes GBD WebSuite Projekt enthält eine *Karte* (``map``), die eine Sammlung von *Layern* (``layers``) ist. Es gibt verschiedene Arten von Ebenen (z. B. "Qgis" oder "WMS"). Sie können Zugriffsrechte, Ansichtseigenschaften (wie ein Extent) und die Metadaten für die gesamte Karte und für jede Ebene individuell konfigurieren. Die meisten Ebenen sind auch an *Quellen* gebunden, die dem Server mitteilen, woher die Geodaten stammen. Eine Layer-Konfiguration enthält typischerweise Anweisungen für den Server, wie die Quelldaten transformiert werden. In der aktuellen Version unterstützt GWS folgende Geodaten-Quellen:

- PostGIS Tabellen
- WMS/WMTS und WFS Dienste
- Kacheldienste wie Open Street Map
- GeoJSON

%info
 In der Zukunft sind auch Rasterquellen, Shape und Geopackage Daten geplant.
%end

%info
 ^config/map und ^config/layer.
%end

Suche und Features
------------------

In der GBD WebSuite sind die Funktionen wie Suche nach dem Schlüsselwort oder auch räumliche Suche durch Klicken oder Ziehen einheitlich *Suche* (``search``) genannt. Es können diverse Such-Quellen (``provider``) konfiguriert werden.

Ein Feature ist ein Objekt das sowohl Sachdaten in Form von *Attributen*, als auch Geoinformation in Form einer *Geometrie* enthält. Die Suchergebnisse sind, unabhängig von der Art der Suche, als eine Liste von Features repräsentiert.

GWS bietet Werkzeuge um die Features aus diversen Quellen im Client oder in einem OWS Dienst einheitlich darzustellen. Dazu gehören *Datenmodellen* (``dataModel``), die Attributen transformieren und *Vorlagen* (``template``), die aus Attributen Präsentationsobjekte, wie HTML Snippets, erstellen.

%info
 ^config/search und ^config/feature.
%end

Arbeiten mit QGIS
-----------------

Die GBD WebSuite bietet dedizierten Support für `QGIS <https://qgis.org>`_, ein kostenloses und quelloffenes geografisches Informationssystem. Die Unterstützung ist optional und kann abgeschaltet werden, wenn Sie QGIS nicht verwenden.

QGIS Projekte können in den GWS Karten reibungslos integriert werden. Ein QGIS Projekt wird als ein Layer in der GWS Karte dargestellt und kann mit anderen Layer-Typen frei kombiniert werden.

Für Drucken unterstützt GWS auch die QGIS Druckvorlagen ("Layouts"), die auch für nicht-QGIS Karten verwendet werden können.

%info
 Mehr dazu in ^config/qgis.
%end
