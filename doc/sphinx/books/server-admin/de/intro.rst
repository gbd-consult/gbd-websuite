Einführung
==========

Was ist die GBD WebSuite
------------------------

Bei der GBD WebSuite handelt es sich um einen Anwendungs- und Webserver, welcher den Schwerpunkt auf die Geodatenverarbeitung legt.

Die GBD WebSuite als Webserver:

- kann statische- und Template-Inhalte bedienen
- unterstützt Multi-Site-Konfigurationen, URL-Routing und Rewriting
- unterstützt verschiedene Berechtigungsmechanismen (Dateisystem, Datenbank, LDAP) und feinkörnige Berechtigungen

Die GBD WebSuite als Geo-Server:

- kombiniert verschiedene Quellen (WMS, Kachelserver, Datenbanken) zu einer einheitlichen Karte
- hat direkte Unterstützung für QGIS-Projekte
- Caches, Reprojekte und Skalierung von Rasterdaten nach Bedarf kann Vektordaten verarbeiten und rendern (PostGIS, Shapes, JSON)
- bietet OGC-konforme Dienste an (WMS, WMTS, WFS)

Die GBD WebSuite als Anwendungsserver:

- bietet einen Rahmen für domänenspezifische Erweiterungen
- hat eine modulare Architektur für einfache Integration

Die GBD WebSuite ist eine Docker-Anwendung, die die folgenden Komponenten beinhaltet:

- `NGINX <https://www.nginx.com/>`_ Webserver, der für statische Inhalte sowie URL-Rewriting zuständig ist
- `uWSGI <https://github.com/unbit/uwsgi>`_ Application Server, der dynamische Anfragen bearbeitet
- Python Komponente ("Aktionen"), die für Datenabfragen und Verarbeitung zuständig sind
- `QGIS <https://qgis.org>`_ Server, zum Rendern von QGIS Projekten
- `MapProxy <https://mapproxy.org/>`_ Server, zum Cachen von Kartenbildern

Open Source
-----------

Die GBD WebSuite basiert vollständig auf Free Open Source Software, und ist eine frei zugängliche Software (Apache License 2.0).

Sponsoren und Mitwirkende
-------------------------

Wir danken unseren Sponsoren...

Wie ist dieses Buch aufgebaut
-----------------------------

Wenn Sie die GBD WebSuite zum ersten Mal nutzen, starten Sie mit dem Kapitel ^quickstart, welches den ersten Start des Servers und Ihres ersten Projektes beschreibt. Im Kapitel ^install wird die Installation der GBD WebSuite ausführlicher beschrieben.  In dem Kapitel ^concepts werden dann die Grundkonzepte und Funktionen der GBD WebSuite vorgestellt. Im Kapitel ^config/index finden Sie detaillierte Anweisungen zur Konfiguration. In dem Kapitel ^ref/index finden Sie eine Auflistung aler Konfigurationsoptionen sowie aller Kommandozeilen Tools.
