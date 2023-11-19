## Allgemeiner Überblick :/common-de

Die GBD WebSuite ist eine Open Source WebGIS Plattform zur Geodatenverarbeitung und wird seit 2017 entwickelt. Sie ist unter der GNU Affero General Public License ([AGPL](https://www.gnu.org/licenses/agpl-3.0.de.html)) lizensiert und wird über Docker Compose plattformunabhängig in bestehende IT-Infrastrukturen integriert.

Die GBD WebSuite ist eine Serveranwendung und stellt einen eigenen, responsiven WebGIS Client bereit, der individuell an eine bestehende Umgebung angepasst oder in diese integriert werden kann. 

Nahezu alle Funktionalität ist plugin-basiert. So können kleine, individuelle Tools für bestimmte Installationen und Zwecke bereitgestellt werden. 

![](uebersicht.png)

Die GBD WebSuite als {title Webserver}

  *  kann statische Inhalte und Vorlagen bereitstellen
  *  unterstützt multi-site Konfigurationen, url routing und rewriting
  *  unterstützt verschiedene Anmelde- und Autorisierungsmechanismen (File System, Datenbank, LDAP) und bietet ein feinabgestimmtes Rechtemanagement.

Die GBD WebSuite als {title Kartenserver}

  *  kombiniert unterschiedliche Quellen (WMS, Tile Server und Datenbanken) in einer Karte
  *  ermöglicht das direkte Einbindung von QGIS Projekten (QGS-, QGZ-Format) auch aus PostgreSQL 
  *  bietet Cachen, Reprojizieren und Resampling von Rasterdaten
  *  erlaubt das direkte Verarbeiten und Darstellen von Vektordaten (z.B. PostGIS, Shapefile, JSON)
  *  ermöglicht die Integration und die Bereitstellung von OGC Diensten (z.B. WMS, WMTS, WFS)

Die GBD WebSuite als {title Applikationsserver}

  *  bietet ein Framework für domänenspezifische Erweiterungen
  *  verfügt über eine modulare Architektur zur einfachen Integration von (Fach)-Anwendungen

Weitere Informationen, die Software und den Quellcode finden Sie auf der [GBD WebSuite Projektseite](https://gbd-websuite.de/).

### :/common-de/funktionalitaet
### :/common-de/komponenten
### :/common-de/gbd
%comment
### :/common-de/lizenz
%end

