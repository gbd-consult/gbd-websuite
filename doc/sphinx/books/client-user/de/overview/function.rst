GBD WebSuite
============

Funktionalität
..............


Die **GBD WebSuite** bietet eine Vielzahl an Funktionalitäten, welche auf Basis von Plugins zur Verfügung gestellt werden und sich fortlaufend erweitern:

* QGIS und QGIS Server Integration als Plugin, welches nativ mit QGIS kommuniziert.
* Integration von Hintergrundkarten
* Abfragen und Darstellen per Klick und Mouseover für WMS
* Auswahl und Abfrage von WFS
* Authentifizierung (LDAP und PostgreSQl)
* Digitalisieren von Punkten, Linien und Flächen
* ALKIS Integration, sowie Suche von und Beauskunftung über Flurstücken
* Schnittstelle zur Fachanwendung D-ProCon (Demographie)
* Schnittstelle zur Fachanwendung GeKoS Online (E-Government)
* Schnittstellen zu anderen GIS (Mapnik, ArcGIS)
* Konfigurierbare Suchmaschinen (Nominatim, ALKIS-Adressen und PostGIS)
* Drucken von Zeichnungen, die im GBD WebSuite Client erzeugt werden
* Caching von Getmap und GetPrint Anfragen



GBD WebSuite Server und Client
..............................


Der **GBD WebSuite Server** basiert auf Python mit Support für PyQGIS und weiteren GIS Bibliotheken. Er läuft in einem WSGI container hinter einem Caching Nginx Proxy.

Der **GBD WebSuite Client** basiert auf React JS und verwendet als Kartenbibliothek OpenLayers. Er besitzt ein responsive Design, das sich jeweils auf die Bildschirmauflösung des Clients einstellt. Um ein optimales Zusammenspiel mit OpenLayers zu erreichen und um die Erweiterung, Wartung und Pflege zu optimieren wurde ein UI Framework implementiert. Die UI Bibliothek wird über einen Abstraktionslayer angesprochen, sodass diese aktualisiert oder ersetzt werden kann, ohne den Kern der Plattform zu beeinflussen.

Da nahezu alle Server- und Client-Funktionalität Plugin-basiert ist, können sehr kleine, angepasste Tools für bestimmte Installationen und Zwecke bereitgestellt werden. Selbst die Integration von QGIS Funktionalitäten ist als Plugin realisiert. Dabei besteht die Möglichkeit, dass QGIS Desktop Benutzer über ein Plugin, Anpassungen an der Konfiguration von Server und Client vornehmen können.
