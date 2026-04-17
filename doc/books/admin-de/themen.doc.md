# Themen :/admin-de/themen

Hier finden Sie Konfigurationsbeispiele und Erklärungen zu verschiedenen Funktionen und Einstellungen, die Sie in der GBD WebSuite konfigurieren können:

[Abfrage](/admin-de/plugin/abfrage)  
[Auswahl](/admin-de/plugin/auswahl)  
[Authentifizierung & Autorisierung](/admin-de/themen/auth)  
[Bemaßung](/admin-de/plugin/dimension)  
[Client](/admin-de/config/client)  
[Drucken](/admin-de/themen/printer)  
[Editieren](/admin-de/themen/edit)  
[Karten](/admin-de/themen/karten)  
[Layer](/admin-de/themen/layer)  <!-- TODO -->  
[Modelle](/admin-de/themen/models)  
[OWS Diesnte](/admin-de/plugin/ows)  
[Plugins](/admin-de/themen/plugins)  <!-- TODO -->  
[PostgreSQL](/admin-de/themen/postgresql)  
[QField](/admin-de/themen/qfield)  
[Projekte](/admin-de/themen/projekte)  <!-- TODO -->  
[Server Konfiguration](/admin-de/config/server)  
[Suche](/admin-de/themen/suche)  
[Vorlagen](/admin-de/config/template)  



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

## Layer


### Raster-Layers
<!--
- qgis,qgisflat,tile,wms,wmsflat,wmts
- caching (/admin-de/layer/cache)
-->

### Vector-Layers
<!--
- geojson,postgres,wfs,??wfsflat??
- link to styling
- link to editing & models
-->

### Flat/Tree/Group Layers
<!--
- group geojson postgres qgis qgisflat tile wfs wfsflat wms wmsflat wmts
- clientOptions, autoLayers
-->

## :/admin-de/config/models
<!--
- (maybe include in editing?)
-->

## :/admin-de/plugin/ows


## Plugins
<!--
- usage and including of plugins, not plugin development
-->

### alkis
### gekos
### :/admin-de/plugin/qgis

<!--
QGIS Client Einstellungen
-->
### select <--- is this useful on it's own?
### annotate
### dimension

## :/admin-de/themen/postgresql

## :/admin-de/themen/qfield



## Projekte
<!--
- uids, metadata and usage in assets/index.cx.html
- project-level templates
- inclusion via projects vs projectDirs vs projectPaths
- overriding global configuration for assets, actions, client.xxx, ...
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


