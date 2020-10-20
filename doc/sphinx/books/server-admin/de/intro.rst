Einführung
==========

Was ist die GBD WebSuite
------------------------

Die GBD WebSuite ist ein Anwendungs-Webserver mit Schwerpunkt Geodatenverarbeitung.

Die GBD WebSuite als Webserver:

- kann statische und Template-Inhalte bedienen
- unterstützt Multi-Site-Konfigurationen, URL-Routing und Rewriting
- unterstützt verschiedene Berechtigungsmechanismen (Dateisystem, Datenbank, LDAP) und feinkörnige Berechtigungen

Die GBD WebSuite als Geo-Server:

- kombiniert verschiedene Quellen (WMS, Kachelserver, Datenbanken) zu einer einheitlichen Karte
- hat direkte Unterstützung für QGIS-Projekte
- Caches, Reprojekte und Skalierung von Rasterdaten nach Bedarf kann Vektordaten verarbeiten und rendern (PostGIS, Shapes, JSON)
- bietet OGC-konforme Dienste an (WMS, WMTS, WFS)

Die GBD WebSuite als Anwendungsserver:

- bietet einen Rahmen für domänenspezifische Erweiterungen
- hat eine steckbare Architektur für einfache Integration lll

Die GBD WebSuite ist eine Docker-Anwendung, die folgende Komponente beinhaltet:

- `NGINX <https://www.nginx.com/>`_ Webserver, der für statische Inhalte sowie URL-Rewriting zuständig ist
- `uWSGI <https://github.com/unbit/uwsgi>`_ Application Server, der dynamische Anfragen bearbeitet
- Python Komponente ("Aktionen"), die für Datenabfragen und Verarbeitung zuständig sind
- `QGIS <https://qgis.org>`_ Server, zum Rendern von QGIS Projekten
- `MapProxy <https://mapproxy.org/>`_ Server, zum Cachen von Kartenbildern

Open Source
-----------

Die GBD WebSuite basiert vollständig auf Free Open Source Software, und ist eine frei zugängliche Software (MIT Lizenz).

Sponsoren und Mitwirkende
-------------------------

Wir danken unsere Sponsoren...

Wie ist dieses Buch aufgebaut
-----------------------------

Wenn Sie neu zu GWS sind, starten Sie mit dem Kapitel ^quickstart und dann ^concepts, wo Sie eine Überblick der GWS Funktionen und Grundkonzepte  finden. Kapitel ^config/index enthält detaillierte Anweisungen zur Konfiguration. Unter ^ref/index finden Sie eine Auflistung aller Konfigurationsoptionen sowie Kommandozeile Tools.
