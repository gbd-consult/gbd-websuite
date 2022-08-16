Übersicht
=========

Die **GBD WebSuite** ist eine webbasierte Open Source GIS Plattform zur Geodatenverarbeitung.
Sie beinhaltet den GBD WebSuite Server und GBD WebSuite Client und zeichnet sich neben der klassischen WebGIS Funktionalität vor allem dadurch aus,
externe Anwendungen und neue Funktionalitäten modular und effizient zu integrieren und umfangreiche Konfigurationen zu ermöglichen.
Die Kern-Bibliotheken von Client und Server sind kompakt gehalten. Die gesamte Architektur ist Plugin-basiert.
Auch die QGIS Integration ist als Plugin implementiert, welche nativ mit QGIS kommuniziert.

Die aktuelle Funktionalität umfasst:

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

.. rubric:: GBD WebSuite Server und Client

Der **GBD WebSuite Server** basiert auf Python mit Support für PyQGIS und weiteren GIS Bibliotheken.
Er läuft in einem WSGI Container, hinter einem Caching NGINX Proxy.

Der **GBD WebSuite Client** basiert auf React JS und verwendet als Kartenbibliothek OpenLayers.
Er besitzt ein responsive Design, das sich jeweils auf die Bildschirmauflösung des Clients einstellt.
Um ein optimales Zusammenspiel mit OpenLayers zu erreichen und um die Erweiterung, Wartung und Pflege zu optimieren, wurde ein UI Framework implementiert.
Die UI Bibliothek wird über einen Abstraktionslayer angesprochen, sodass diese aktualisiert oder ersetzt werden kann, ohne den Kern der Plattform zu beeinflussen.

Da nahezu alle Server- und Client-Funktionalitäten Plugin-basiert sind, können sehr kleine, angepasste Tools für bestimmte Installationen und Zwecke bereitgestellt werden.
Selbst die Integration von QGIS Funktionalitäten ist als Plugin realisiert.
Dabei besteht die Möglichkeit, dass QGIS Desktop Benutzer über ein Plugin, Anpassungen an der Konfiguration von Server und Client vornehmen können.

Die GBD WebSuite als **Web Server**:

  *  kann statische Inhalte und Vorlagen bereitstellen
  *  unterstützt multi-site Konfigurationen, url routing und rewriting
  *  unterstützt verschiedene Anmelde- und Autorisierungsmechanismen (File System, Datenbank, LDAP und Postgres) und bietet ein feinabgestimmtes Rechtemanagement.

Die GBD WebSuite als **Geo Server**:

  *  kombiniert unterschiedliche Quellen (WMS, Tile Server und Datenbanken) in einer Karte
  *  ermöglicht die direkte Einbindung von QGIS Projekten (QGS-Format, QGZ-Format)
  *  bietet Cachen, Reprojizieren und Resampling von Rasterdaten
  *  erlaubt das direkte Verarbeiten und Rendern von Vektordaten (PostGIS, Shapefile, JSON)
  *  ermöglicht die Integration von OGC Diensten (WMS, WMTS, WFS)

Die GBD WebSuite als **Applikations Server**:

  *  bietet ein Framework für domänenspezifische Erweiterungen
  *  verfügt über eine modulare Architektur zur einfachen Integration von (Fach)-Anwendungen

.. toctree::
   :maxdepth: 3

   how_to_use.rst
   gbd.rst
.. function.rst
