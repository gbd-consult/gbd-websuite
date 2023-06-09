# GBD WebSuite Documentation :/


Die **GBD WebSuite** ist eine webbasierte Open Source GIS Plattform zur Geodatenverarbeitung.
Sie beinhaltet den GBD WebSuite Server und GBD WebSuite Client und zeichnet sich neben der klassischen WebGIS Funktionalität vor allem dadurch aus, externe Anwendungen und neue Funktionalitäten modular und effizient zu integrieren und umfangreiche Konfigurationen zu ermöglichen.
Die Kern-Bibliotheken von Client und Server sind schlank gehalten. Die gesamte Architektur ist Plugin-basiert.
Auch die QGIS Integration ist als Plugin implementiert, welche nativ mit QGIS kommuniziert.

**Geschichte**

Das Projekt wurde im Herbst 2017 begonnen. Am 10. Januar 2019 wurde die Version 1.0.0 veröffentlicht und über github und dockerhub bereitgestellt. Seit dieser Zeit arbeiten wir hart daran, die **GBD WebSuite** zu verbessern und mit neuer Funktionalität zu erweitern.

**Lizenz und Veröffentlichung**

Die **GBD WebSuite** wird als Docker Image bereitgestellt und kann plattformunabhängig in IT-Infrastrukturen integriert werden. Sie kombiniert die Funktionalität zahlreicher Open Source Software, wie QGIS, MapProxy, uWSGI oder NGINX und stellt diese den Anwendern zur Verfügung.

Die **GBD WebSuite** wird unter der GNU Affero General Public License (AGPL) veröffentlicht.


**Funktionalität** **Funktionsübersicht**

Sie zeichnet sich neben der klassischen WebGIS Funktionalität vor allem dadurch aus, externe Anwendungen und
neue Funktionalitäten modular und effizient zu integrieren und umfangreiche Konfigurationen zu ermöglichen. Die
Kern-Bibliotheken von Client und Server sind schlank gehalten. Die gesamte Architektur ist Plugin-basiert. Auch die QGIS
Integration ist als Plugin implementiert, welche nativ mit QGIS kommuniziert.

Dem Anwender stellt die **GBD WebSuite** eine Vielzahl an GIS Funktionalität durch Kernfunktionen und Erweiterungen zur Verfügung, um Informationen in der Karte zu suchen, abzufragen, zu erstellen, zu bearbeiten und als druckfertige Karte auszugeben. Externe (Fach-)Anwendungen und neue Funktionalität können modular integriert und deren Nutzung individuell konfiguriert werden.

Die **GBD WebSuite** bietet eine Vielzahl an Funktionalitäten, welche auf Basis von Plugins zur Verfügung gestellt und flexibel konfiguriert werden können.
Das Aussehen des GBD WebSuite Clients kann individuell an eine bestehende Umgebung angepasst oder in diese integriert werden. Die aktuelle Funktionalität umfasst:

* QGIS und QGIS Server Integration
* Caching von Getmap und GetPrint Anfragen
* Einbinden von Hintergrunddiensten
* Objektabfragen per Klick oder Mouseover
* Räumliche Suche von Objekten in der Karte
* Markieren und Messen in der Karte
* Bemaßung von Segmentlängen in der Karte
* Benutzer-Authentifizierung (LDAP-, Postgres- und Datei-basiert)
* Editieren von Punkten, Linien und Flächen
* ALKIS Integration (Flurstücksuche, Beauskunftung und Druck)
* Konfigurierbare Suche (Nominatim, ALKIS-Adressen und Postgres)
* Schnittstelle zur Fachanwendung D-ProCon (Demographische Analysen)
* Schnittstelle zur Fachanwendung GekoS Online (E-Government)
* Drucken in unterschiedlicher Auflösung inklusive redlining
* Screenshots erstellen im PNG-Format

GBD WebSuite Server und Client
..............................

Der **GBD WebSuite Server** basiert auf Python mit Support für PyQGIS und weiteren GIS Bibliotheken.
Er läuft in einem WSGI Container hinter einem Caching NGINX Proxy.

Der **GBD WebSuite Client** basiert auf React JS und verwendet als Kartenbibliothek OpenLayers.
Er besitzt ein responsive Design, das sich jeweils auf die Bildschirmauflösung des Clients einstellt.
Um ein optimales Zusammenspiel mit OpenLayers zu erreichen und um die Erweiterung, Wartung und Pflege zu optimieren wurde ein UI Framework implementiert.
Die UI Bibliothek wird über einen Abstraktionslayer angesprochen, sodass diese aktualisiert oder ersetzt werden kann, ohne den Kern der Plattform zu beeinflussen.

Da nahezu alle Server- und Client-Funktionalitäten Plugin-basiert sind, können sehr kleine, angepasste Tools für bestimmte Installationen und Zwecke bereitgestellt werden.
Selbst die Integration von QGIS Funktionalitäten ist als Plugin realisiert.
Dabei besteht die Möglichkeit, dass QGIS Desktop Benutzer über ein Plugin, Anpassungen an der Konfiguration von Server und Client vornehmen können.

Die **GBD WebSuite** ist eine modulare Open Source WebGIS Plattform zur Geodatenverarbeitun. Demzufolge verwendet diese für die Darstellung von Daten in der Karte WebGIS Layer und WebGIS Gruppen.
Um in der nachfolgenden Anleitung kurze Namen verwenden zu können, folgt eine Übersicht von verwendeten Abkürzungen.


Die GBD WebSuite als **Web Server**:

  *  kann statische Inhalte und Vorlagen bereitstellen
  *  unterstützt multi-site Konfigurationen, url routing und rewriting
  *  unterstützt verschiedene Anmelde- und Autorisierungsmechanismen (File System, Datenbank, LDAP) und bietet ein feinabgestimmtes Rechtemanagement.

Die GBD WebSuite als **Geo Server**:

  *  kombiniert unterschiedliche Quellen (WMS, Tile Server und Datenbanken) in einer Karte
  *  ermöglicht die direkte Einbindung von QGIS Projekten (QGS-Format, QGZ-Format)
  *  bietet Cachen, Reprojizieren und Resampling von Rasterdaten
  *  erlaubt das direkte Verarbeiten und Rendern von Vektordaten (PostGIS, Shapefile, JSON)
  *  ermöglicht die Integration von OGC Diensten (WMS, WMTS, WFS)

Die GBD WebSuite als **Applikations Server**:

  *  bietet ein Framework für domänenspezifische Erweiterungen
  *  verfügt über eine modulare Architektur zur einfachen Integration von (Fach)-Anwendungen


**Hinweise zur Dokumentation**

In diesem Benutzerhandbuch werden zum besseren Verständnis verschiedene Schreibstile verwendet.


| Formatierung          | Beschreibung                                      |
|-----------------------|---------------------------------------------------|
| {title Bedienelement} | Titel des Bedienelements                          |
| {button Schältfläche} | Schaltfläche zum Anklicken                        |
| {param Parameter}     | Parameter mit Eingabemöglichkeit                  |
| {link Verlinkung}     | innerhalb des Handbuchs oder zu externer Webseite |

Weitere Informationen finden sie im:

%toc
/user-de
/admin-de
/dev-en
%end


## :/user-de/gbd
## :/websuite-manager 
## :/user-de
## :/admin-de
## :/dev-en