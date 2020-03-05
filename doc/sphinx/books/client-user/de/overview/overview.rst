Übersicht
=========

Die **GBD WebSuite** ist eine webbasierte Open Source GIS Plattform zur Geodatenverarbeitung. Sie beinhaltet den GBD WebSuite Server und GBD WebSuite Client und zeichnet sich neben der klassischen WebGIS Funktionalität vor allem dadurch aus, externe Anwendungen und neue Funktionalitäten modular und effizient zu integrieren und umfangreiche Konfigurationen zu ermöglichen. Die Kern-Bibliotheken von Client und Server sind schlank gehalten. Die gesamte Architektur ist Plugin-basiert. Auch die QGIS Integration ist als Plugin implementiert, welche nativ mit QGIS kommuniziert.

Die GBD WebSuite als **Web Server**:

  *  kann statische Inhalte und Vorlagen bereitstellen
  *  unterstützt multi-site Konfigurationen, url routing und rewriting
  *  unterstützt verschiedene Anmelde- und Autorisierungsmechanismen (File System, Datenbank, LDAP) und bietet ein feinabgestimmtes Rechtemanagement.

Die GBD WebSuite als **Geo Server**:

  *  kombiniert unterschiedliche Quellen (WMS, Tile Server und Datenbanken) in einer Karte
  *  ermöglicht das direkte Einbindung von QGIS Projekten
  *  bietet Cachen, Reprojizieren und Resampling von Rasterdaten
  *  erlaubt das direkte Verarbeiten und Rendern von Vektordaten (PostGIS, Shapefile, JSON)
  *  ermöglicht die Integration von OGC Diensten (WMS, WMTS, WFS)

Die GBD WebSuite als **Applikations Server**:

  *  bietet ein Framework für domänenspezifische Erweiterungen
  *  verfügt über eine modulare Architektur zur einfachen Integration von (Fach)-Anwendungen

Die GBD WebSuite wird als Docker Image bereitgestellt und kann plattformunabhängig in IT-Infrastrukturen integriert werden. Sie kombiniert die Funktionalität zahlreicher Open Source Software, wie QGIS, MapProxy, uWSGI oder NGINX und stellt diese den Anwendern zur Verfügung.


.. toctree::
   :maxdepth: 2

   function.rst
   gbd.rst
