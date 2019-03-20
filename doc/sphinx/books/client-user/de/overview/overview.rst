Übersicht
=========

Die **GBD WebSuite** ist eine webbasierte Open Source GIS Plattform zur Geodatenverarbeitung. Sie beinhaltet den GBD WebSuite Server und GBD WebSuite Client und zeichnet sich neben der klassischen WebGIS Funktionalität vor allem dadurch aus, externe Anwendungen und neue Funktionalitäten modular und effizient zu integrieren und umfangreiche Konfigurationen zu ermöglichen. Die Kern-Bibliotheken von Client und Server sind schlank gehalten. Die gesamte Architektur ist Plugin-basiert. Auch die QGIS Integration ist als Plugin implementiert, welche nativ mit QGIS kommuniziert.

GBD WebSuite als Web Server:

  *  kann statische Inhalte und Vorlagen bereitstellen
  *  unterstützt multi-site Konfigurationen, url routing und rewriting
  *  unterstützt verschiedene Anmelde- und Autorisierungsmechanismen (File System, Datenbank, LDAP) und fein abgestimmte Rechtevergaben


GBD WebSuite als Geo Server:

  *  kombiniert unterschiedliche Quellen (WMS, Tile Server und Datenbanken) in einer Karte
  *  direkte Einbindung von QGIS Projekten
  *  Rasterdaten können gecached, reprojektiert und skaliert werden
  *  Vektordaten können verarbeitet und gerendert werden (PostGIS, Shapefile, Json)
  *  ermöglicht die Integration von OGC Diensten (WMS, WMTS, WFS)


GBD WebSuite als Applikations Server:

  *  bietet ein Framework für domänenspezifische Erweiterungen
  *  verfügt über eine modulare Architektur zur einfachen Integration

GBD WebSuite ist programmiert mit Hilfe von Open Source Software, wie:

  *  QGIS
  *  MapProxy
  *  uWSGI
  *  NGINX


========================================================================================================================================================================================

.. toctree::
    :maxdepth: 2

    function.rst
    gbd.rst
