Übersicht
=========

Die **GBD WebSuite** ist eine webbasierte Open Source GIS Plattform zur Geodatenverarbeitung.
Sie ermöglicht das Editieren und Visualisieren von Geodaten.
QGIS Projekte können auf bequeme Art bereitgestellt und online bearbeitet werden.
Des Weiteren können externe Anwendungen und neue Funktionalitäten modular und effizient integriert und konfiguriert werden.

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
