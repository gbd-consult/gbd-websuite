## Allgemeiner Überblick :/common-de

Die GBD WebSuite ist eine Open Source WebGIS Plattform zur Geodatenverarbeitung. Sie wird seit 2017 entwickelt, ist unter der GNU Affero General Public License ([AGPL](https://www.gnu.org/licenses/agpl-3.0.de.html)) lizensiert und wird mittels Docker Compose plattformunabhängig in bestehende IT-Infrastrukturen integriert. Sie ist eine Serveranwendung und stellt einen eigenen, responsiven WebGIS Client bereit, der individuell an eine bestehende Umgebung angepasst oder in diese integriert werden kann. 

**Übersicht zur Funktionalität**

Nahezu alle Funktionalität ist plugin-basiert. So können individuelle Anwendungen für bestimmte Installationen und Zwecke entwickelt und bereitgestellt werden.

{title GBD WebSuite als Webserver}

  *  kann statische Inhalte und Vorlagen bereitstellen
  *  unterstützt multi-site Konfigurationen, url routing und rewriting
  *  unterstützt verschiedene Anmelde- und Autorisierungsmechanismen (File System, Datenbank, LDAP) und bietet ein feinabgestimmtes Rechtemanagement.

{title GBD WebSuite als Kartenserver}

  *  kombiniert unterschiedlichste Quellen (Dateien, OGC Dienste und Datenbanken) in einer Karte
  *  unterstützt das direkte Einbinden von QGIS Projekten (QGS-, QGZ-Format) auch aus PostgreSQL 
  *  bietet Cachen und Reprojizieren von Geodaten und das Resampling von Rasterdaten
  *  erlaubt das Editieren und Darstellen von Vektordaten (z.B. PostGIS, Shapefile, JSON)
  *  ermöglicht die Bereitstellung von OGC- und INSPIRE-Diensten (z.B. als WMS, WMTS, WFS)

{title GBD WebSuite als Applikationsserver}

  *  bietet ein Framework für domänenspezifische Erweiterungen
  *  verfügt über eine modulare Architektur zur einfachen Integration von (Fach)-Anwendungen

Neben dieser Dokumentation bietet die [GBD WebSuite Homepage](https://gbd-websuite.de/) weitere, detaillierte Informationen zum Projekt mit zahlreichen Anwendungsbeispielen, sowie Verlinkungen zum Download der Software und zum Quellcode.

### :/common-de/funktionalitaet
### :/common-de/komponenten
### :/common-de/gbd
%comment
### :/common-de/lizenz
%end

