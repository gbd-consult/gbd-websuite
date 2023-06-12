## Allgemein :/common-de



### Geoinformatikbüro Dassau GmbH :/common-de/gbd

Die **Geoinformatikbüro Dassau GmbH** ist seit mehr als 10 Jahren an der Entwicklung von Open Source Software und Projekten beteiligt. Wir setzen diese Erfahrung erfolgreich in GIS- und WebGIS-Projekten, sowie in Geodateninfrastrukturen (GDI) in Deutschland und darüber hinaus, bei unseren Kunden ein.

Auf Basis unserer Expertise bieten wir umfassende Unterstützung in den Bereichen GIS und GDI. Dazu gehört:

* Beratung und Konzeption
* Schulung und Workshops
* Datenverarbeitung und Fernerkundung
* Softwaredesign und -entwicklung
* Wartung und Support

Unsere Kernkompetenz liegt in der Software QGIS Desktop, QGIS Server, QGIS Web Client, GRASS GIS und PostgreSQL/POSTGIS und der von uns entwickelten GBD WebSuite.

Weitere Informationen finden Sie auf unserer [Website](https://www.gbd-consult.de).

### Was ist die GBD WebSuite :/common-de/websuite

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

- [NGINX](https://www.nginx.com) Webserver, der für statische Inhalte sowie URL-Rewriting zuständig ist
- [uWSGI](https://github.com/unbit/uwsgi) Application Server, der dynamische Anfragen bearbeitet
- Python Komponente ("Aktionen"), die für Datenabfragen und Verarbeitung zuständig sind
- [QGIS](https://qgis.org) Server, zum Rendern von QGIS Projekten
- [MapProxy](https://mapproxy.org) Server, zum Cachen von Kartenbildern


### Lizenz und Veröffentlichung :/common-de/lizenz

Die **GBD WebSuite** wird als Docker Image bereitgestellt und kann plattformunabhängig in IT-Infrastrukturen integriert werden. Sie kombiniert die Funktionalität zahlreicher Open Source Software, wie QGIS, MapProxy, uWSGI oder NGINX und stellt diese den Anwendern zur Verfügung.


Die GBD WebSuite basiert vollständig auf Free Open Source Software, und wird unter der GNU Affero General Public License (AGPL) veröffentlicht.

Den Quellcode finden sie auf [GitHub](https://github.com/gbd-consult/gbd-websuite)

### Geschichte :/common-de/geschichte

Das Projekt wurde im Herbst 2017 begonnen. Am 10. Januar 2019 wurde die Version 1.0.0 veröffentlicht und über github und dockerhub bereitgestellt. Seit dieser Zeit arbeiten wir hart daran, die **GBD WebSuite** zu verbessern und mit neuer Funktionalität zu erweitern.