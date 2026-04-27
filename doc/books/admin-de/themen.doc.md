# Themen :/admin-de/themen

Hier finden Sie Konfigurationsbeispiele und Erklärungen zu verschiedenen Funktionen und Einstellungen, die Sie in der GBD WebSuite konfigurieren können:

- [**Abfrage**](/admin-de/plugin/abfrage)  
  Konfiguriert das Identify-Tool, mit dem Benutzer durch Klicken oder Hovern Objektinformationen aus verschiedenen Layer-Typen abrufen und über Vorlagen darstellen können.

- [**Auswahl**](/admin-de/plugin/auswahl)  
  Beschreibt das Auswahl-Tool zum Markieren und Speichern mehrerer Kartenobjekte durch Klicken oder Polygon-Zeichnen.

- [**Authentifizierung & Autorisierung**](/admin-de/themen/auth)  
  Erklärt das Zugriffssystem der GBD WebSuite mit Authentifizierungsprovidern (Datei, LDAP, PostgreSQL) und rollenbasierter Zugriffssteuerung über ACL-Strings.

- [**Bemaßung**](/admin-de/plugin/dimension)  
  Das Bemaßungs-Tool erlaubt das Zeichnen von Maßlinien mit optionalem Einrasten an Vektorobjekten aus PostgreSQL-Layern.

- [**Client**](/admin-de/config/client)  
  Beschreibt die Konfiguration des Browser-Clients mit UI-Elementen (Toolbar, Sidebar, Infobar), Anzeigeoptionen und rollenbasierter Elementsteuerung.

- [**Drucken**](/admin-de/themen/printer)  
  Zeigt die Konfiguration von Druckvorlagen (HTML und QGIS) mit verschiedenen Qualitätsstufen sowie speziellen Template-Tags für Karte, Legende und Seitenstruktur.

- [**Editieren**](/admin-de/themen/edit)  
  Erklärt das Konfigurieren von editierbaren PostgreSQL-Layern für das Zeichnen und Bearbeiten von Vektorobjekten mit Attributen.

- [**Karten**](/admin-de/themen/karten)  
  Beschreibt die Grundkonfiguration einer Karte: Koordinatenreferenzsystem, Ausdehnung, Mittelpunkt, Zoomstufen und Layer-Reihenfolge für Haupt- und Übersichtskarte.

- [**Layer**](/admin-de/themen/layer)
  Gibt einen Überblick über alle Layer-Typen – Raster (QGIS, WMS, WMTS, Tile), Vektor (GeoJSON, PostgreSQL, WFS) sowie Gruppen- und Flat/Tree-Varianten mit Caching- und Client-Optionen.

- [**Modelle**](/admin-de/themen/models)  
  Datenmodelle strukturieren Feature-Attribute aus verschiedenen Quellen für Anzeige, Bearbeitung und Druckvorlagen mittels regelbasierter Transformationen, Feldeditoren und Validatoren.

- [**OWS Dienste**](/admin-de/plugin/ows)  
  Beschreibt die Bereitstellung von OGC-Webdiensten (WMS, WFS, WCS, WMTS, CSW) mit Layer-spezifischer Konfiguration, URL-Rewriting und INSPIRE-Unterstützung.

- [**Plugins**](/admin-de/themen/plugins)  <!-- TODO -->  
  Erklärt die Verwendung und Einbindung von Plugins in GBD WebSuite-Projekte.

- [**PostgreSQL**](/admin-de/themen/postgresql)  
  Beschreibt die Konfiguration von PostgreSQL-Datenbankverbindungen mit Unterstützung für `pg_service.conf`, automatisches Schema-Caching und Projektablage in der Datenbank.

- [**QField**](/admin-de/themen/qfield)  
  Beschreibt die Integration von QField für mobile Felddatenerfassung.

- [**Projekte**](/admin-de/themen/projekte)  <!-- TODO -->  
  <!-- Erklärt Projektkonfiguration mit UIDs, Metadaten, Einbindungsmethoden (`projects`, `projectDirs`, `projectPaths`) und projektspezifischen Überschreibungen für Assets, Aktionen und Client-Einstellungen. -->

- [**Server Konfiguration**](/admin-de/config/server)  
  Beschreibt serverweite Einstellungen wie Logging-Level, Module (web, mapproxy, qgis, spool, monitor) mit Worker-Konfiguration sowie die Datenablage für benutzerdefinierte Objekte.

- [**Suche**](/admin-de/themen/suche)  
  Zeigt die Konfiguration des Suchsystems mit Findern für verschiedene Quellen (Nominatim, PostgreSQL, WFS, WMS) und die Darstellung der Ergebnisse über Vorlagen.

- [**Vorlagen**](/admin-de/config/template)  
  Erklärt die Template-Engine mit `@`-basierter Steuersprache für Text-, HTML-, XML-, Karten- und Python-Vorlagen für Webseiten, Infoboxen, Druckdokumente und OWS-Dienste.



## :/admin-de/plugin/abfrage

<!--
 In File: /gbd-websuite/app/gws/plugin/identify_tool/_doc/admin_de.doc.md
- info
- spatial info
-->

## :/admin-de/plugin/auswahl
<!--
 In File: /gbd-websuite/app/gws/plugin/select_tool/_doc/admin_de.doc.md
- select
- mouseover select 
-->

## :/admin-de/themen/auth
<!--
 In File: /gbd-websuite/app/gws/base/auth/_doc/admin-de.doc.md
- Preface: Authentication vs Authorization
- Authentication: file, postgres, ldap(ldap could be own topic)
- Authorization: access, permissions
-->

## :/admin-de/plugin/dimension

## :/admin-de/config/client
<!--
- client elements
- clientOptions
- including in project.cx.html
-->

## :/admin-de/themen/printer
<!--
 In File: /gbd-websuite/app/gws/base/printer/_doc/admin-de.doc.md
- templates
    - html, map, py, qgis
- link to models
- qualityLevels
-->

## :/admin-de/themen/edit
<!--
- edit layers
    - link to styling
- link to models
    - attr. table
-->


## :/admin-de/themen/karten
<!--
- map vs overviewMap
- crs, center, extent, extentBuffer, zoom
- layers: ordering, link to layer thema
-->

## :/admin-de/themen/layer


<!-- ### Raster-Layers -->
<!--
- qgis,qgisflat,tile,wms,wmsflat,wmts
- caching (/admin-de/layer/cache)
-->

<!-- ### Vector-Layers -->
<!--
- geojson,postgres,wfs,??wfsflat??
- link to styling
- link to editing & models
-->

<!-- ### Flat/Tree/Group Layers -->
<!--
- group geojson postgres qgis qgisflat tile wfs wfsflat wms wmsflat wmts
- clientOptions, autoLayers
-->

## :/admin-de/config/models
<!--
- (maybe include in editing?)
-->

## :/admin-de/plugin/ows


## :/admin-de/themen/plugins
<!--
- usage and including of plugins, not plugin development
-->
<!-- 
### alkis
### gekos -->

<!-- ### select <--- is this useful on it's own?
### annotate
### dimension -->

## :/admin-de/themen/projekte
<!--
- uids, metadata and usage in assets/index.cx.html
- project-level templates
- inclusion via projects vs projectDirs vs projectPaths
- overriding global configuration for assets, actions, client.xxx, ...
-->

## :/admin-de/themen/postgresql

## :/admin-de/themen/qfield

### :/admin-de/plugin/qgis

<!--
QGIS Client Einstellungen
-->



## :/admin-de/config/server
<!--
- qgis server specific things?
- fonts?
- server stuff:
    - developer options
    - helpers? maybe include in other topics, as csv and xml helpers are weird topics on their own.
    - storage (maybe include in other topics where applicable)
    - web.*
    - server.*
-->

## :/admin-de/themen/suche


## :/admin-de/config/template
<!--
- maybe include in various places? general templating? single vs double curly braces?
-->


