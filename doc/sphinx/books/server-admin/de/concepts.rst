Grundkonzepte
=============

In diesem Kapitel beschreiben wir kurz, auf welche Grundkonzepten die Funktionalität von GWS basiert ist.

Anfragen und URLs
-----------------

Einmal gestartet, hört die GBD WebSuite auf Ports ``80/443`` und verarbeitet ``GET`` und ``POST`` Anfragen. Wie ein herkömmlicher Webserver kann GWS statische Inhalte wie HTML-Seiten oder Bilder bereitstellen, aber sein Hauptzweck ist es, dynamische Kartenbilder und Daten bereitzustellen. Für dynamische Anfragen gibt es einen einzigen Endpunkt (url), nämlich den ``_`` (Unterstrich). Alle Anfragen an diesen Endpunkt müssen den Befehl (``cmd``) Parameter enthalten.
Zusätzlich müssen alle ``POST`` Anfragen im JSON-Format vorliegen

Hier ein paar Beispiele von Anfragen, die GBD WebSuite bearbeiten kann.

Eine statische GET-Anfrage: ::

    http://maps.my-server.com/images/smile.jpg

Eine dynamische GET-Anfrage (z. B. ein Kartenbild): ::

    http://maps.my-server.com/_?cmd=mapHttpGetBox&projectUid=london&layerUid=london.map.metro&width=100&height=200&bbox=10,20,30,40

Eine dynamische POST-Anfrage (z.B. Suche): ::

    {
        "cmd":"search",
        "params": {
            "projectUid": "london",
            "bbox": [10,20,30,40],
            "keyword": "queen"
        }
    }

GET-URLs können durch URL-Rewriting modifiziert werden, so dass diese URL ::

    http://maps.my-server.com/_?cmd=mapHttpGetBox&projectUid=london&layerUid=london.map.metro

zu dieser reduziert werden kann ::

    http://maps.my-server.com/london/metro

^SEE Eine ausführliche Anleitung zur URL-Rewriting find Sie unter ^config/web

Aktionen
--------

Anhand vom ``cmd`` Parameter entscheidet der Server welche *Server Aktion* die Bearbeitung der Anfrage übernimmt. Falls die Aktion existiert und richtig konfiguriert ist,  wird die Anfrage zu dieser Aktion weitergeleitet. Die Aktion bearbeitet die Anfrage und stellt eine Antwort bereit, die abhängig von der Natur der Anfrage, in HTML, JSON oder PNG Format vorliegt. Intern sind die Aktionen die Python-Klassen, die für jeden Befehl (``cmd``) über eine Methode verfügen. Im obigen Beispiel (``cmd=mapHttpGetBox``) ist ``map`` die Aktion und ``httpGetBox`` die Methode die diese Anfrage bearbeitet.

^SEE Server Aktionen sind unter ^config/actions beschrieben

Webseiten und Projekte
----------------------

Auf der obersten Ebene arbeitet die GBD WebSuite mit zwei Arten von Entitäten: *Projekte* und *Webseiten*. Ein Projekt ist in etwa eine Karte und eine Sammlung von Einstellungen, die sich auf diese spezielle Karte beziehen. Eine Site ist ein Domain-Name, der an Autorisierungs- und Routing-Regeln gebunden ist.

In dieser URL ::

    http://maps.my-server.com/_?cmd=mapHttpGetBox&projectUid=london&layerUid=london.map.metro

ist ``london`` ein Projekt, ``london.map.metro`` ist eine für dieses Projekt konfigurierte Ebene, während der Domainname ``maps. my-server. com`` und die entsprechende Rewrite-Regel aus der Webseiten-Konfiguration übernommen werden.

Webseiten und Projekte sind orthogonale Konzepte. Sie können dasselbe Projekt unter mehreren Webseiten ausführen. Wenn Sie z. B. ``maps.my-server.com`` to e.g. ``gis.my-other-server.com`` ändern würden, würde dies keine Änderungen im Projekt ``london`` erfordern.

Statische Dokumente und Assets
------------------------------

Autorisierung
-------------

GWS enthält eine Rollen-Basierte Autorisierung. Bei allen konfigurierbaren Systemobjekte kann mittels *Zugriffsblöcken* (``access``) spezifiziert werden, welche Rollen den Zugriff zu diesem Objekt haben. Falls es für die gegebene Rolle keine explizite Anweisung gibt, wird das übergeordnete Objekte gecheckt. Für das Root Objekt (`application`) werden per default alle Zugriffe verweigert.

Individuelle Zugangsdaten (Nutzername, Passwort) werden zu Rollen mittels *Autorisierungsanbieter* (``provider``) verknüpft. Die Aufgabe eines Providers ist, die Zugangsdaten gegen der angegebenen Datenquelle zu prüfen. Aktuell werden folgende Provider unterstützt:

* LDAP
* file-basiert

^NOTE In der Zukunft sind auch Datenbank Provider geplant.

*Autorisierungsmethoden* (``method``) geben an, wie die Zugangsdaten dem System übergeben werden. Aktuell sind diese Methoden unterstützt:

- ``web``: Übergabe mittels eines Web-Formulars (Login-Form)
- ``basic``: Übergabe mittels einer HTTP-Basic Autorisierung

^NOTE In der Zukunft sind auch OAuth, Two-Factor sowie Windows single sign-on (SSO) geplant.

^SEE Mehr dazu in ^config/auth.

Karten und Layer
----------------

Jedes GBD WebSuite Projekt enthält eine *Karte* (``map``), die eine Sammlung von *Layern* (``layers``) ist. Es gibt verschiedene Arten von Ebenen (z. B. "Qgis" oder "WMS"). Sie können Zugriffsrechte, Ansichtseigenschaften (wie ein Extent) und die Metadaten für die gesamte Karte und für jede Ebene individuell konfigurieren. Die meisten Ebenen sind auch an *Quellen* gebunden, die dem Server mitteilen, woher die Geodaten stammen. Eine Layer-Konfiguration enthält typischerweise Anweisungen für den Server, wie die Quelldaten transformiert werden. In der aktuellen Version unterstützt GWS folgende Geodaten-Quellen:

* PostGIS Tabellen
* WMS/WMTS und WFS Dienste
* Kacheldienste wie Open Street Map
* GeoJSON

^NOTE In der Zukunft sind auch Rasterquellen, Shape und Geopackage Daten geplant.

Suche und Features
------------------

In GWS sind die Funktionen wie Suche nach dem Schlüsselwort oder auch räumliche Suche durch Klicken oder Ziehen einheitlich *Suche* (``search``) genannt. Es können diverse Such-Quellen (``provider``) konfiguriert werden.

Ein Feature ist ein Objekt das sowohl Sachdaten in Form von *Attributen*, als auch Geoinformation in Form einer *Shape* enthält. Die Suchergebnisse sind, unabhängig von der Art der Suche, als eine Liste von Features repräsentiert.

GWS bietet Werkzeuge um die Feautres aus diversen Quellen einheitlich im Client oder in einem OWS Dienst einheitlich darzustellen. Dazu gehören *Datenmodellen* (``dataModel``), die Attributen transformieren und *Vorlagen* (``template``), die aus Attributen Präsentiationsobjekte, wie HTML Snippets, erstellen.

^SEE config/features

Arbeiten mit QGIS
-----------------

QGIS Projekte können in den GWS Maps reibungslos integriert werden.

^SEE ^config/qgis
