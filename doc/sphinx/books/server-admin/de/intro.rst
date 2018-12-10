Einführung
==========


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
- hat eine steckbare Architektur für einfache Integration

Die GBD WebSuite basiert vollständig auf Free Open Source Software, um nur einige zu nennen: 

- `QGIS <https://qgis.org>`_
- `MapProxy <https://mapproxy.org/>`_
- `uWSGI <https://github.com/unbit/uwsgi>`_
- `NGINX <https://www.nginx.com/>`_

und ist eine frei zugängliche Software.


