# i18n Migration Conflicts

Generated while consolidating `app/gws/spec/strings.de.ini.alt` into per-module `_doc/strings.ini`.

## Summary

- identical (skipped): **16**
- merged into module files: **145**
- conflicts (different German text in module file vs. .alt): **32**
- orphans (no module owns this UID): **384**

## Conflicts (manual review)

For each entry below the module-local German text was kept; the `.alt` text is shown for reference.

### `gws.Config.uid`
- file: `app/gws/core/_doc/strings.ini`
- module: ```
Objekt-ID.
```
- .alt: ```
Einzigartige ID, optional, außer wenn zur Referenz nötig
```

### `gws.base.application.core.Config`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Hauptkonfiguration enthält globale Einstellungen, sowie die Projektkonfigurationen. Einige Konfigurationen wie `actions`, `finders`, `models` und `templates` können in den Projekten erweitert werden. Die Projekte werden direkt unter `projects` oder über Dateipfade `projectPaths` bzw. `projectDirs` geladen.
```
- .alt: ```
Hauptkonfiguration der GBD WebSuite
```

### `gws.base.application.core.Config.actions`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Serveraktionen definieren serverseitige Funktionalität. Sie stellen die Schnittstelle zwischen Frontend (Client) und Backend (Server) dar. Über Actions werden grundlegende Funktionen wie `project` (Projektkonfiguration), `map` (Kartendarstellung), `search` (Datenabfragen), `edit` (Feature-Bearbeitung) oder `print` (Kartenausgabe) bereitgestellt.
```
- .alt: ```
verfügbare Serveraktionen
```

### `gws.base.application.core.Config.auth`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Autorisierungsmethoden und -optionen steuern den Zugriff auf GBD WebSuite Ressourcen. Hier werden Provider wie `system` (lokale Benutzer), `ldap` (Active Directory-Integration), `sql` (datenbankbasierte Authentifizierung) oder `web` (externe Webdienste) konfiguriert. Über Sessions, Rollen und Berechtigungen wird granular gesteuert, welche Benutzer auf welche Projekte, Layer und Funktionen zugreifen dürfen.
```
- .alt: ```
Autorisierungsmethoden und -optionen
```

### `gws.base.application.core.Config.cache`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Cache Konfiguration definiert die Zwischenspeicherung von Daten und gerenderten Inhalten zur Performance-Optimierung. Hier werden Cache-Typen wie `redis` (In-Memory-Datenbank), `file` (dateisystembasiert) oder `memory` (Arbeitsspeicher) konfiguriert. Der Cache beschleunigt wiederkehrende Anfragen für Kartenkacheln, WMS-Responses, Suchergebnisse und Metadaten und reduziert die Serverlast bei gleichzeitiger Verbesserung der Antwortzeiten.
```
- .alt: ```
Cache Konfiguration
```

### `gws.base.application.core.Config.client`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Client Konfiguration steuert das Verhalten und Aussehen der Client-Benutzeroberfläche. Hier werden UI-Elemente wie Toolbar, Sidebar, Map oder Infobar sowie deren Anordnung und Funktionalitäten konfiguriert. Die Client-Konfiguration steuert Themes, verfügbare Tools, Kartenwerkzeuge und die allgemeine Benutzererfahrung. Dadurch lassen sich projektspezifische Oberflächen erstellen, die genau auf die Anforderungen der jeweiligen Anwendergruppe zugeschnitten sind.
```
- .alt: ```
GBD WebSuite Client Konfiguration
```

### `gws.base.application.core.Config.database`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Datenbank Konfiguration definiert die Verbindungen zu externen Datenbanksystemen. Hier werden Provider wie `postgres` (PostgreSQL/PostGIS), `sqlite` (lokale Dateien) oder andere SQL-Datenbanken mit Verbindungsparametern, Credentials und Performance-Einstellungen konfiguriert. Die Database-Konfiguration ermöglicht den Zugriff auf Geodaten, Attributtabellen, Benutzerdaten und Metadaten aus verschiedenen Datenquellen für Layer, Authentifizierung und Geschäftslogik.
```
- .alt: ```
Datenbank Konfiguration
```

### `gws.base.application.core.Config.developer`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Entwickler Konfiguration definiert Einstellungen für Debugging, Entwicklung und erweiterte Systemdiagnose der GBD WebSuite. Hier werden Optionen wie `reloader` (automatische Code-Aktualisierung), `profiler` (Performance-Analyse), Logging-Level und Debug-Modi konfiguriert. Die Konfiguration aktiviert erweiterte Fehlerausgaben, Entwicklerwerkzeuge und Monitoring-Funktionen, die bei der Systementwicklung, Fehlerbehebung und Performance-Optimierung unterstützen.
```
- .alt: ```
Entwickler-Optionen
```

### `gws.base.application.core.Config.finders`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Such- und Abfrage-Konfiguration definiert die verschiedenen Suchmechanismen und Abfragefunktionalitäten. Hier werden Suchtypen wie `keyword` (Textsuche), `geometry` (räumliche Suche), `filter` (attributbasierte Abfragen) oder `postgres` (datenbankspezifische Suchen) konfiguriert. Die Finders-Konfiguration ermöglicht es Benutzern, gezielt nach Features, Adressen, Koordinaten oder fachlichen Inhalten zu suchen und steuert die Darstellung der Suchergebnisse in der Client-Oberfläche.
```
- .alt: ```
Konfiguration der Such-Anbieter
```

### `gws.base.application.core.Config.fonts`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die globale Konfiguration für Fonts definiert die Schriftarten für die Darstellung von Text in Karten, Labels und der Benutzeroberfläche. Hier werden Schriftfamilien wie Arial, Helvetica oder benutzerdefinierte Fonts mit ihren Pfaden und Eigenschaften konfiguriert. Die Fonts-Konfiguration steuert die Textdarstellung in Kartenbeschriftungen, Feature-Labels, Print-Ausgaben und Client-Elementen und ermöglicht ein einheitliches Corporate Design sowie die Unterstützung verschiedener Sprachen und Zeichensätze.
```
- .alt: ```
Konfiguration der Schriftart
```

### `gws.base.application.core.Config.helpers`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Hilfsdienste und Erweiterungen definieren zusätzliche Dienste und Funktionalitäten, die andere GBD WebSuite-Komponenten unterstützen. Hier werden spezialisierte Services wie `alkis` (Katasterdatenverarbeitung), `csv` (Datenimport/-export), `ows` (OGC-Webservice-Integration) oder `storage` (Dateiverwaltung) konfiguriert. Die Helpers-Konfiguration erweitert die Grundfunktionalität um fachspezifische Werkzeuge, Datenkonverter und externe Service-Integrationen, die von Layern, Server Aktionen und anderen Modulen genutzt werden können.
```
- .alt: ```
Konfiguration der Helfer
```

### `gws.base.application.core.Config.locales`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Internationalisierung und Sprachkonfiguration definiert die verfügbaren Sprachen und regionalen Einstellungen für die GBD WebSuite Benutzeroberfläche. Hier werden Sprachcodes wie `de_DE` (Deutsch), `en_GB` (Englisch) oder `cs_CZ` (Tschechisch) mit ihren Übersetzungsdateien und Formatierungsregeln konfiguriert. Die Locales-Konfiguration steuert die Lokalisierung von Menüs, Dialogen, Fehlermeldungen und Datumsformaten und ermöglicht mehrsprachige Anwendungen für internationale Benutzergruppen.
```
- .alt: ```
Konfiguration der Sprache
```

### `gws.base.application.core.Config.metadata`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Metadaten-Konfiguration definiert beschreibende Informationen über das GBD WebSuite Projekt und dessen Inhalte. Hier werden Eigenschaften wie `title` (Projekttitel), `abstract` (Beschreibung), `keywords` (Schlagwörter), `contact` (Kontaktinformationen) und `inspireMandatory` (INSPIRE-konforme Metadaten) konfiguriert. Die Metadata-Konfiguration stellt projektbezogene Informationen für Katalogdienste, OGC-Services und Compliance-Anforderungen bereit und unterstützt die Auffindbarkeit und Dokumentation von Geodatenbeständen.
```
- .alt: ```
Konfiguration der Metadaten
```

### `gws.base.application.core.Config.models`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Datenmodell-Konfiguration definiert die Struktur und Eigenschaften von Datenobjekten in der GBD WebSuite. Hier werden Feldtypen wie `text`, `int`, `geometry` oder `date`, Validierungsregeln, Beziehungen zwischen Objekten und Darstellungsoptionen konfiguriert. Die Models-Konfiguration steuert die Datenvalidierung, Feature-Editierung, Such- und Anzeigelogik und bildet die Grundlage für typisierte Geodatenobjekte, Formulare und datenbankbasierte Layer.
```
- .alt: ```
Konfiguration für Modelle
```

### `gws.base.application.core.Config.owsServices`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die OGC-Webservice-Konfiguration definiert die von der GBD WebSuite bereitgestellten standardkonformen OGC-Webservices. Hier werden Services wie `wms` (Web Map Service), `wfs` (Web Feature Service), `wmts` (Web Map Tile Service) oder `csw` (Catalog Service Web) mit ihren Capabilities, Layern und Zugriffsrechten konfiguriert. Die owsServices-Konfiguration ermöglicht die Bereitstellung von Geodaten über standardisierte Schnittstellen für externe GIS-Anwendungen, Desktop-Clients und andere GBD WebSuite Instanzen.
```
- .alt: ```
Konfiguration der Open Web Services
```

### `gws.base.application.core.Config.printers`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Druck-Konfiguration definiert die verfügbaren Druckoptionen und -formate für Kartenausgaben der GBD WebSuite. Hier werden Druckvorlagen mit Eigenschaften wie `pageSize` (Seitenformat), `orientation` (Ausrichtung), `dpi` (Auflösung), Template-Layouts und Ausgabeformate wie PDF oder PNG konfiguriert. Die Printers-Konfiguration ermöglicht hochqualitative Kartendrucke mit Legenden, Maßstäben, Nordpfeilen und benutzerdefinierten Layouts für professionelle Kartenpräsentationen und Berichte.
```
- .alt: ```
Konfigurationen fürs Drucken
```

### `gws.base.application.core.Config.projectDirs`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Projektverzeichnis-Konfiguration definiert die Dateisystempfade, in denen GBD WebSuite-Projekte und deren Ressourcen gespeichert werden. Hier werden Verzeichnisse für Projektkonfigurationen, QGIS-Dateien, Geodaten, Templates und Asset-Dateien mit ihren absoluten oder relativen Pfaden konfiguriert. Die projectDirs-Konfiguration strukturiert die Dateiorganisation, ermöglicht die zentrale Verwaltung von Projektressourcen und steuert die Zugriffsrechte auf verschiedene Datenbestände für eine saubere Trennung von Entwicklungs-, Test- und Produktionsumgebungen.
```
- .alt: ```
Verzeichnisse mit zusätzlichen Projekten
```

### `gws.base.application.core.Config.projectPaths`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Einzelprojekt-Pfadkonfiguration definiert den spezifischen Dateisystempfad zu einer einzelnen Projektkonfigurationsdatei oder einem Projektverzeichnis. Hier wird der absolute oder relative Pfad zur .cx-Konfigurationsdatei oder zum Projektordner angegeben, der das GBD WebSuite-Projekt enthält. Die projectPath-Konfiguration ermöglicht die direkte Referenzierung eines bestimmten Projekts und wird typischerweise in Single-Project-Installationen oder bei der gezielten Aktivierung einzelner Projekte aus einem größeren Projektbestand verwendet.
```
- .alt: ```
zusätzliche Projektpfade
```

### `gws.base.application.core.Config.projects`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Mehrprojekt-Konfiguration definiert eine Liste von GBD WebSuite-Projekten, die gleichzeitig auf dem Server verfügbar sind. Hier werden mehrere Projekte mit ihren jeweiligen Konfigurationen, UIDs, Pfaden und spezifischen Einstellungen definiert. Die projects-Konfiguration ermöglicht Multi-Tenant-Umgebungen, bei denen verschiedene Anwendungen, Fachbereiche oder Organisationseinheiten separate Projekte mit individuellen Karten, Layern und Berechtigungen nutzen können, während sie auf derselben WebSuite-Instanz betrieben werden.
```
- .alt: ```
Projektkonfigurationen
```

### `gws.base.application.core.Config.server`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Server-Konfiguration definiert die grundlegenden HTTP-Server-Einstellungen und Netzwerkparameter der GBD WebSuite-Instanz. Hier werden Eigenschaften wie `host` (Bind-Adresse), `port` (Listening-Port), `timeout` (Request-Timeouts), SSL-Zertifikate, Worker-Prozesse und Performance-Parameter konfiguriert. Die server-Konfiguration steuert die Netzwerkverfügbarkeit, Sicherheitseinstellungen, Lastverteilung und Skalierbarkeit der GBD WebSuite und bildet die technische Grundlage für alle Client-Server-Kommunikation und externe Service-Zugriffe.
```
- .alt: ```
Optionen für den GBD WebSuite Server
```

### `gws.base.application.core.Config.storage`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Speicher-Konfiguration definiert die Datenspeicherung und Dateiverwaltung für temporäre und persistente Inhalte der WebSuite. Hier werden Speicherorte für Upload-Dateien, Session-Daten, Cache-Inhalte, Benutzer-Assets und temporäre Verarbeitungsergebnisse mit ihren Pfaden, Größenbeschränkungen und Aufbewahrungszeiten konfiguriert. Die storage-Konfiguration steuert die Dateisystem-Organisation, Speicherplatz-Management und Cleanup-Routinen und ermöglicht die sichere Verwaltung von Benutzerdaten, Import-/Export-Dateien und systeminternen Arbeitsdateien.
```
- .alt: ```
Optionen zum Speichern der Konfiguration
```

### `gws.base.application.core.Config.templates`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Template-Konfiguration definiert die Vorlagen für die dynamische Generierung von HTML-Inhalten, Popups und Ausgabeformaten in der GBD WebSuite. Hier werden Template-Typen wie `html` (Feature-Popups), `text` (Textausgaben), `xml` (strukturierte Daten) oder `css` (Styling) mit ihren Pfaden, Variablen und Formatierungsregeln konfiguriert. Die templates-Konfiguration ermöglicht die Anpassung von Feature-Informationen, Suchergebnissen, Print-Layouts und Client-Darstellungen durch wiederverwendbare Vorlagen mit dynamischen Inhalten und projektspezifischem Design.
```
- .alt: ```
Konfiguration von Vorlagen
```

### `gws.base.application.core.Config.web`
- file: `app/gws/base/application/_doc/strings.ini`
- module: ```
Die Web-Client-Konfiguration definiert die Bereitstellung und das Verhalten der Browser-basierten Benutzeroberfläche der GBD WebSuite. Hier werden Eigenschaften wie `root` (URL-Pfad), `cors` (Cross-Origin-Einstellungen), `ssl` (HTTPS-Konfiguration), `assets` (statische Ressourcen) und `rewrite` (URL-Umschreibungsregeln) konfiguriert. Die web-Konfiguration steuert die Client-Auslieferung, Sicherheitsrichtlinien, Asset-Verwaltung und URL-Routing und bildet die Grundlage für den Zugriff auf GBD WebSuite-Anwendungen über Webbrowser und externe Client-Integrationen.
```
- .alt: ```
Konfiguration des Webservers
```

### `gws.base.auth.manager.Config.methods`
- file: `app/gws/base/auth/_doc/strings.ini`
- module: ```
Definition verfügbarer Autorisierungsmethoden wie Basic Auth, OAuth, LDAP oder Token-basierte Verfahren.
```
- .alt: ```
Methoden der Authentifizierung
```

### `gws.base.auth.manager.Config.mfa`
- file: `app/gws/base/auth/_doc/strings.ini`
- module: ```
Konfiguration von Multi-Faktor-Authentifizierungsmethoden für erhöhte Sicherheitsanforderungen.
```
- .alt: ```
Multi-Faktor-Authentifizierung
```

### `gws.base.auth.manager.Config.providers`
- file: `app/gws/base/auth/_doc/strings.ini`
- module: ```
Sammlung verschiedener Autorisierungsanbieter für externe Authentifizierungsquellen und Identity Provider.
```
- .alt: ```
Anbieter für die Authentifizierung
```

### `gws.base.auth.manager.Config.session`
- file: `app/gws/base/auth/_doc/strings.ini`
- module: ```
Umfassende Sitzungsverwaltungsoptionen für Timeout, Persistierung und Session-Sicherheit.
```
- .alt: ```
Sitzungs-Optionen
```

### `gws.base.auth.provider.Config.allowedMethods`
- file: `app/gws/base/auth/_doc/strings.ini`
- module: ```
Whitelist erlaubter Autorisierungsmethoden für spezifische Provider-Konfigurationen.
```
- .alt: ```
Optionales Einschränken der Authentifizierungsmethoden, falls dieser Provider nicht für alle Methoden gültig sein soll.
```

### `gws.base.database.model.Config.dbUid`
- file: `app/gws/base/database/_doc/strings.ini`
- module: ```
Zugewiesene UID einer spezifischen Datenbankverbindung für Datenmodell-Operationen.
```
- .alt: ```
Eindeutige Beschreibung des Datenbank Providers
```

### `gws.base.web.manager.Config.ssl`
- file: `app/gws/base/web/_doc/strings.ini`
- module: ```
Zentrale SSL/TLS-Konfiguration für sichere HTTPS-Verbindungen und Zertifikatsverwaltung.
```
- .alt: ```
SSL-Konfiguration
```

### `gws.base.web.site.Config.canonicalHost`
- file: `app/gws/base/web/_doc/strings.ini`
- module: ```
Kanonischer Hostname für SEO-optimierte URL-Generierung und Reverse-Proxy-Unterstützung.
```
- .alt: ```
Hostname für reverse rewriting
```

### `gws.base.web.site.Config.root`
- file: `app/gws/base/web/_doc/strings.ini`
- module: ```
Dokumenten-Stammverzeichnis für statische Web-Inhalte und File-Serving.
```
- .alt: ```
Speicherort und Optionen des Dokumentstammverzeichnisses
```

## Orphans

Entries from `.alt` whose UID prefix did not match any module:

- `cli.auth.test` = `Interaktives Testen 'einer' "Anmeldung".`
- `cli.server.reconfigure` = `Konfigurieren Sie den Server neu und laden Sie ihn neu`
- `cli.server.reload` = `Laden Sie den Server neu, ohne ihn neu zu konfigurieren`
- `cli.server.start` = `Starten Sie den GBD WebSuite Server`
- `cli.server.stop` = `Stoppen Sie den GBD WebSuite Server`
- `gws.SourceLayerFilter` = `Layerfilter`
- `gws.auth.types.Config` = `Authentifizierungs- und Autorisierungsoptionen`
- `gws.auth.types.Config.cookie` = `Parameter für Sitzungscookies`
- `gws.auth.types.Config.httpEnabled` = `HTTP-Autorisierung aktiviert`
- `gws.auth.types.Config.httpsOnly` = `HTTP-Autorisierung nur über SSL`
- `gws.auth.types.Config.providers` = `Autorisierungsanbieter`
- `gws.auth.types.Config.session` = `Sitzungskonfiguration`
- `gws.auth.types.CookieConfig` = `Parameter für Sitzungscookies`
- `gws.auth.types.CookieConfig.name` = `Name für den Cookie`
- `gws.auth.types.CookieConfig.path` = `Cookie-Pfad`
- `gws.auth.types.SessionConfig` = `Sitzungskonfiguration`
- `gws.auth.types.SessionConfig.heartBeat` = `Sitzungen automatisch aktualisieren`
- `gws.auth.types.SessionConfig.lifeTime` = `Sitzungslebenszeit`
- `gws.auth.types.SessionConfig.storage` = `Sitzungs-Speicher-Engine`
- `gws.ext.action.alkis.Config` = `Flurstücksuche`
- `gws.ext.action.alkis.Config.access` = `Zugriffsrechte`
- `gws.ext.action.alkis.Config.alkisSchema` = `Schema, in dem ALKIS-Tabellen gespeichert sind, muss lesbar sein`
- `gws.ext.action.alkis.Config.db` = `Datenbank (PostGIS) Provider-ID`
- `gws.ext.action.alkis.Config.eigentuemerAccess` = `Zugriff auf die Eigentümer-Informationen`
- `gws.ext.action.alkis.Config.excludeGemarkung` = `Gemarkung (AU) IDs, die von der Suche ausgeschlossen werden sollen`
- `gws.ext.action.alkis.Config.featureFormat` = `Vorlage für Flurstück-Details auf dem Bildschirm`
- `gws.ext.action.alkis.Config.grundbuchAccess` = `Zugriff auf die Grundbuch-Informationen`
- `gws.ext.action.alkis.Config.indexSchema` = `Schema zum Speichern der internen Indizes der GBD WebSuite, muss beschreibbar sein`
- `gws.ext.action.alkis.Config.limit` = `Suchergebnisse begrenzen`
- `gws.ext.action.alkis.Config.model` = `Flurstück-Modell`
- `gws.ext.action.alkis.Config.printTemplate` = `Vorlage für gedruckte Flurstück-Details`
- `gws.ext.action.alkis.Config.type` = `Objekttyp`
- `gws.ext.action.asset.Config` = `Asset (dynamische HTML) Aktion)`
- `gws.ext.action.asset.Config.access` = `Zugriffsrechte`
- `gws.ext.action.asset.Config.type` = `Objekttyp`
- `gws.ext.action.auth.Config` = `Autorisierungsaktion`
- `gws.ext.action.auth.Config.access` = `Zugriffsrechte`
- `gws.ext.action.auth.Config.type` = `Objekttyp`
- `gws.ext.action.edit.Config` = `feature edit action`
- `gws.ext.action.edit.Config.access` = `Zugriffsrechte`
- `gws.ext.action.edit.Config.type` = `Objekttyp`
- `gws.ext.action.map.Config` = `map rendering action`
- `gws.ext.action.map.Config.access` = `Zugriffsrechte`
- `gws.ext.action.map.Config.type` = `Objekttyp`
- `gws.ext.action.print.Config` = `Druckaktion`
- `gws.ext.action.print.Config.access` = `Zugriffsrechte`
- `gws.ext.action.print.Config.type` = `Objekttyp`
- `gws.ext.action.project.Config` = `Projektaktion`
- `gws.ext.action.project.Config.access` = `Zugriffsrechte`
- `gws.ext.action.project.Config.type` = `Objekttyp`
- `gws.ext.action.remoteadmin.Config` = `Remote-Administratoraktion`
- `gws.ext.action.remoteadmin.Config.access` = `Zugriffsrechte`
- `gws.ext.action.remoteadmin.Config.type` = `Objekttyp`
- `gws.ext.action.search.Config` = `Suchaktion`
- `gws.ext.action.search.Config.Grenze` = `search results limit`
- `gws.ext.action.search.Config.access` = `Zugriffsrechte`
- `gws.ext.action.search.Config.pixelTolerance` = `Pixeltoleranz für Geometriesuche`
- `gws.ext.action.search.Config.type` = `Objekttyp`
- `gws.ext.db.provider.postgis.Config` = `PostGIS-Datenbankanbieter`
- `gws.ext.db.provider.postgis.Config.database` = `Name der Datenbank`
- `gws.ext.db.provider.postgis.Config.host` = `Datenbankhost`
- `gws.ext.db.provider.postgis.Config.password` = `Passwort`
- `gws.ext.db.provider.postgis.Config.port` = `Datenbankport`
- `gws.ext.db.provider.postgis.Config.timeout` = `Abfragezeitüberschreitung`
- `gws.ext.db.provider.postgis.Config.type` = `Objekttyp`
- `gws.ext.db.provider.postgis.Config.uid` = `Eindeutige ID`
- `gws.ext.db.provider.postgis.Config.user` = `Benutzername`
- `gws.ext.layer.box.Config` = `Box-Layer`
- `gws.ext.layer.box.Config.access` = `Zugriffsrechte`
- `gws.ext.layer.box.Config.cache` = `Cache-Konfiguration`
- `gws.ext.layer.box.Config.clientOptions` = `Optionen für die Layeranzeige im Client`
- `gws.ext.layer.box.Config.description` = `Vorlage für die Layerbeschreibung`
- `gws.ext.layer.box.Config.editable` = `Dieser Layer kann bearbeitet werden`
- `gws.ext.layer.box.Config.extent` = `Layerausmasse`
- `gws.ext.layer.box.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.layer.box.Config.grid` = `Rasterkonfiguration`
- `gws.ext.layer.box.Config.meta` = `Layermetadaten`
- `gws.ext.layer.box.Config.opacity` = `Layertransparenz`
- `gws.ext.layer.box.Config.search` = `Layer-Suchkonfiguration`
- `gws.ext.layer.box.Config.source` = `Datenquelle für den Layer`
- `gws.ext.layer.box.Config.sourceLayers` = `Zu verwendende Quelllayer`
- `gws.ext.layer.box.Config.title` = `Layertitel`
- `gws.ext.layer.box.Config.type` = `Objekttyp`
- `gws.ext.layer.box.Config.uid` = `Eindeutige Layer-ID`
- `gws.ext.layer.box.Config.zoom` = `Layeraufloesungen und Massstaebe`
- `gws.ext.layer.group.Config` = `Gruppenlayer`
- `gws.ext.layer.group.Config.access` = `Zugriffsrechte`
- `gws.ext.layer.group.Config.cache` = `Cache-Konfiguration`
- `gws.ext.layer.group.Config.clientOptions` = `Optionen für die Layeranzeige im Client`
- `gws.ext.layer.group.Config.description` = `Vorlage für die Layerbeschreibung`
- `gws.ext.layer.group.Config.editable` = `Dieser Layer kann bearbeitet werden`
- `gws.ext.layer.group.Config.extent` = `Layerausmasse`
- `gws.ext.layer.group.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.layer.group.Config.grid` = `Rasterkonfiguration`
- `gws.ext.layer.group.Config.layers` = `Layer in dieser Gruppe`
- `gws.ext.layer.group.Config.meta` = `Layermetadaten`
- `gws.ext.layer.group.Config.opacity` = `Layertransparenz`
- `gws.ext.layer.group.Config.search` = `Layer-Suchkonfiguration`
- `gws.ext.layer.group.Config.source` = `Datenquelle für den Layer`
- `gws.ext.layer.group.Config.sourceLayers` = `Zu verwendende Quelllayer`
- `gws.ext.layer.group.Config.title` = `Layertitel`
- `gws.ext.layer.group.Config.type` = `Objekttyp`
- `gws.ext.layer.group.Config.uid` = `Eindeutige Layer-ID`
- `gws.ext.layer.group.Config.zoom` = `Layeraufloesungen und Massstaebe`
- `gws.ext.layer.leaf.Config` = `Blattlayer`
- `gws.ext.layer.leaf.Config.access` = `Zugriffsrechte`
- `gws.ext.layer.leaf.Config.cache` = `Cache-Konfiguration`
- `gws.ext.layer.leaf.Config.clientOptions` = `Optionen für die Layeranzeige im Client`
- `gws.ext.layer.leaf.Config.description` = `Vorlage für die Layerbeschreibung`
- `gws.ext.layer.leaf.Config.editable` = `Dieser Layer kann bearbeitet werden`
- `gws.ext.layer.leaf.Config.extent` = `Layerausmasse`
- `gws.ext.layer.leaf.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.layer.leaf.Config.grid` = `Rasterkonfiguration`
- `gws.ext.layer.leaf.Config.meta` = `Layermetadaten`
- `gws.ext.layer.leaf.Config.opacity` = `Layertransparenz`
- `gws.ext.layer.leaf.Config.search` = `Layer-Suchkonfiguration`
- `gws.ext.layer.leaf.Config.source` = `Datenquelle für den Layer`
- `gws.ext.layer.leaf.Config.sourceLayers` = `Zu verwendende Quelllayer`
- `gws.ext.layer.leaf.Config.title` = `Layertitel`
- `gws.ext.layer.leaf.Config.type` = `Objekttyp`
- `gws.ext.layer.leaf.Config.uid` = `Eindeutige Layer-ID`
- `gws.ext.layer.leaf.Config.wmsName` = `Diese Layer-ID im WMS-Dienst`
- `gws.ext.layer.leaf.Config.zoom` = `Layeraufloesungen und Massstaebe`
- `gws.ext.layer.osm.Config.access` = `Zugriffsrechte`
- `gws.ext.layer.osm.Config.cache` = `Cache-Konfiguration`
- `gws.ext.layer.osm.Config.clientOptions` = `Optionen für die Layeranzeige im Client`
- `gws.ext.layer.osm.Config.description` = `Vorlage für die Layerbeschreibung`
- `gws.ext.layer.osm.Config.editable` = `Dieser Layer kann bearbeitet werden`
- `gws.ext.layer.osm.Config.extent` = `Layerausmasse`
- `gws.ext.layer.osm.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.layer.osm.Config.grid` = `Rasterkonfiguration`
- `gws.ext.layer.osm.Config.meta` = `Layermetadaten`
- `gws.ext.layer.osm.Config.opacity` = `Layertransparenz`
- `gws.ext.layer.osm.Config.search` = `Layer-Suchkonfiguration`
- `gws.ext.layer.osm.Config.source` = `Datenquelle für den Layer`
- `gws.ext.layer.osm.Config.sourceLayers` = `Zu verwendende Quelllayer`
- `gws.ext.layer.osm.Config.title` = `Layertitel`
- `gws.ext.layer.osm.Config.type` = `Objekttyp`
- `gws.ext.layer.osm.Config.uid` = `Eindeutige Layer-ID`
- `gws.ext.layer.osm.Config.zoom` = `Layeraufloesungen und Massstaebe`
- `gws.ext.layer.qgis.Config` = `automatische QGIS-Layer`
- `gws.ext.layer.qgis.Config.access` = `Zugriffsrechte`
- `gws.ext.layer.qgis.Config.cache` = `Cache-Konfiguration`
- `gws.ext.layer.qgis.Config.clientOptions` = `Optionen für die Layeranzeige im Client`
- `gws.ext.layer.qgis.Config.description` = `Vorlage für die Layerbeschreibung`
- `gws.ext.layer.qgis.Config.directPostgis` = `Wenn true, können Sie direkte PostGIS-Verbindungen für die Suche verwenden`
- `gws.ext.layer.qgis.Config.directServices` = `Wenn true, verwenden Sie direkte Verbindungen zu externen Diensten`
- `gws.ext.layer.qgis.Config.editable` = `Dieser Layer kann bearbeitet werden`
- `gws.ext.layer.qgis.Config.extent` = `Layerausmasse`
- `gws.ext.layer.qgis.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.layer.qgis.Config.grid` = `Rasterkonfiguration`
- `gws.ext.layer.qgis.Config.meta` = `Layermetadaten`
- `gws.ext.layer.qgis.Config.opacity` = `Layertransparenz`
- `gws.ext.layer.qgis.Config.search` = `Layer-Suchkonfiguration`
- `gws.ext.layer.qgis.Config.source` = `Datenquelle für den Layer`
- `gws.ext.layer.qgis.Config.sourceLayers` = `Zu verwendende Quelllayer`
- `gws.ext.layer.qgis.Config.title` = `Layertitel`
- `gws.ext.layer.qgis.Config.type` = `Objekttyp`
- `gws.ext.layer.qgis.Config.uid` = `Eindeutige Layer-ID`
- `gws.ext.layer.qgis.Config.zoom` = `Layeraufloesungen und Massstaebe`
- `gws.ext.layer.tile.Config` = `Titel-Layer`
- `gws.ext.layer.tile.Config.access` = `Zugriffsrechte`
- `gws.ext.layer.tile.Config.cache` = `Cache-Konfiguration`
- `gws.ext.layer.tile.Config.clientOptions` = `Optionen für die Layeranzeige im Client`
- `gws.ext.layer.tile.Config.description` = `Vorlage für die Layerbeschreibung`
- `gws.ext.layer.tile.Config.editable` = `Dieser Layer kann bearbeitet werden`
- `gws.ext.layer.tile.Config.extent` = `Layerausmasse`
- `gws.ext.layer.tile.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.layer.tile.Config.grid` = `Rasterkonfiguration`
- `gws.ext.layer.tile.Config.meta` = `Layermetadaten`
- `gws.ext.layer.tile.Config.opacity` = `Layertransparenz`
- `gws.ext.layer.tile.Config.search` = `Layer-Suchkonfiguration`
- `gws.ext.layer.tile.Config.source` = `Datenquelle für den Layer`
- `gws.ext.layer.tile.Config.sourceLayers` = `Zu verwendende Quelllayer`
- `gws.ext.layer.tile.Config.title` = `Layertitel`
- `gws.ext.layer.tile.Config.type` = `Objekttyp`
- `gws.ext.layer.tile.Config.uid` = `Eindeutige Layer-ID`
- `gws.ext.layer.tile.Config.zoom` = `Layeraufloesungen und Massstaebe`
- `gws.ext.layer.tree.Config` = `Tree-Layer`
- `gws.ext.layer.tree.Config.access` = `Zugriffsrechte`
- `gws.ext.layer.tree.Config.cache` = `Cache-Konfiguration`
- `gws.ext.layer.tree.Config.clientOptions` = `Optionen für die Layeranzeige im Client`
- `gws.ext.layer.tree.Config.description` = `Vorlage für die Layerbeschreibung`
- `gws.ext.layer.tree.Config.editable` = `Dieser Layer kann bearbeitet werden`
- `gws.ext.layer.tree.Config.extent` = `Layerausmasse`
- `gws.ext.layer.tree.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.layer.tree.Config.grid` = `Rasterkonfiguration`
- `gws.ext.layer.tree.Config.meta` = `Layermetadaten`
- `gws.ext.layer.tree.Config.opacity` = `Layertransparenz`
- `gws.ext.layer.tree.Config.search` = `Layer-Suchkonfiguration`
- `gws.ext.layer.tree.Config.source` = `Datenquelle für den Layer`
- `gws.ext.layer.tree.Config.sourceLayers` = `Zu verwendende Quelllayer`
- `gws.ext.layer.tree.Config.title` = `Layertitel`
- `gws.ext.layer.tree.Config.type` = `Objekttyp`
- `gws.ext.layer.tree.Config.uid` = `Eindeutige Layer-ID`
- `gws.ext.layer.tree.Config.zoom` = `Layeraufloesungen und Massstaebe`
- `gws.ext.layer.vector.Config` = `Vector-Layer`
- `gws.ext.layer.vector.Config.access` = `Zugriffsrechte`
- `gws.ext.layer.vector.Config.cache` = `Cache-Konfiguration`
- `gws.ext.layer.vector.Config.clientOptions` = `Optionen für die Layeranzeige im Client`
- `gws.ext.layer.vector.Config.description` = `Vorlage für die Layerbeschreibung`
- `gws.ext.layer.vector.Config.editStyle` = `Stil für zu bearbeitende Objekte`
- `gws.ext.layer.vector.Config.editable` = `Dieser Layer kann bearbeitet werden`
- `gws.ext.layer.vector.Config.extent` = `Layerausmasse`
- `gws.ext.layer.vector.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.layer.vector.Config.grid` = `Rasterkonfiguration`
- `gws.ext.layer.vector.Config.meta` = `Layermetadaten`
- `gws.ext.layer.vector.Config.opacity` = `Layertransparenz`
- `gws.ext.layer.vector.Config.search` = `Layer-Suchkonfiguration`
- `gws.ext.layer.vector.Config.source` = `Datenquelle für den Layer`
- `gws.ext.layer.vector.Config.sourceLayers` = `Zu verwendende Quelllayer`
- `gws.ext.layer.vector.Config.style` = `Objektstil`
- `gws.ext.layer.vector.Config.title` = `Layertitel`
- `gws.ext.layer.vector.Config.type` = `Objekttyp`
- `gws.ext.layer.vector.Config.uid` = `Eindeutige Layer-ID`
- `gws.ext.layer.vector.Config.zoom` = `Layeraufloesungen und Massstaebe`
- `gws.ext.search.provider.nominatim.Config` = `nominatim (OSM) Suchanbieter`
- `gws.ext.search.provider.nominatim.Config.access` = `Zugriffsrechte`
- `gws.ext.search.provider.nominatim.Config.country` = `Land, um die Suche einzuschraenken`
- `gws.ext.search.provider.nominatim.Config.defaultContext` = `Standardmaeßiger raeumlicher Kontext`
- `gws.ext.search.provider.nominatim.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.search.provider.nominatim.Config.language` = `Sprache der Ergebnisseausgabe`
- `gws.ext.search.provider.nominatim.Config.title` = `Titel der Suchergebnisse`
- `gws.ext.search.provider.nominatim.Config.type` = `Objekttyp`
- `gws.ext.search.provider.qgispg.Config` = `QGIS . Postgres-Anbieter fuer automatische Suche`
- `gws.ext.search.provider.qgispg.Config.access` = `Zugriffsrechte`
- `gws.ext.search.provider.qgispg.Config.defaultContext` = `Standardmaeßiger raeumlicher Kontext`
- `gws.ext.search.provider.qgispg.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.search.provider.qgispg.Config.title` = `Titel der Suchergebnisse`
- `gws.ext.search.provider.qgispg.Config.type` = `Objekttyp`
- `gws.ext.search.provider.sql.Config` = `Datenbankbasierte Suche`
- `gws.ext.search.provider.sql.Config.access` = `Zugriffsrechte`
- `gws.ext.search.provider.sql.Config.db` = `Datenbankanbieter UID`
- `gws.ext.search.provider.sql.Config.defaultContext` = `Standardmaeßiger raeumlicher Kontext`
- `gws.ext.search.provider.sql.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.search.provider.sql.Config.sort` = `Sortierausdruck`
- `gws.ext.search.provider.sql.Config.table` = `SQL-Tabellenkonfiguration`
- `gws.ext.search.provider.sql.Config.title` = `Titel der Suchergebnisse`
- `gws.ext.search.provider.sql.Config.type` = `Objekttyp`
- `gws.ext.search.provider.wms.Config` = `WMS-basierte Suche`
- `gws.ext.search.provider.wms.Config.access` = `Zugriffsrechte`
- `gws.ext.search.provider.wms.Config.defaultContext` = `Standardmaeßiger raeumlicher Kontext`
- `gws.ext.search.provider.wms.Config.featureFormat` = `Objekt-Formatierungsoptionen`
- `gws.ext.search.provider.wms.Config.title` = `Titel der Suchergebnisse`
- `gws.ext.search.provider.wms.Config.type` = `Objekttyp`
- `gws.ext.source.geojson.Config` = `GeoJson Quelle`
- `gws.ext.source.geojson.Config.access` = `Zugriffsrechte`
- `gws.ext.source.geojson.Config.crs` = `KBS für diese Quelle`
- `gws.ext.source.geojson.Config.path` = `Pfad zu einer Geojson-Datei`
- `gws.ext.source.geojson.Config.searchAttribute` = `Textsuchattribut`
- `gws.ext.source.geojson.Config.type` = `Objekttyp`
- `gws.ext.source.qgis.Config` = `QGIS Quelle`
- `gws.ext.source.qgis.Config.crs` = `KBS für diese Quelle`
- `gws.ext.source.qgis.Config.extent` = `Layerausmasse`
- `gws.ext.source.qgis.Config.path` = `Pfad zu einer Qgs-Projektdatei`
- `gws.ext.source.qgis.Config.type` = `Objekttyp`
- `gws.ext.source.sql.Config` = `SQL Quelle`
- `gws.ext.source.sql.Config.db` = `Datenbank Provider-ID`
- `gws.ext.source.sql.Config.table` = `SQL-Tabellenkonfiguration`
- `gws.ext.source.sql.Config.type` = `Objekttyp`
- `gws.ext.source.tile.Config` = `Tile Quelle`
- `gws.ext.source.tile.Config.crs` = `KBS für diese Quelle`
- `gws.ext.source.tile.Config.extent` = `Layerausmasse`
- `gws.ext.source.tile.Config.grid` = `Quell-Rasterkonfiguration`
- `gws.ext.source.tile.Config.type` = `Objekttyp`
- `gws.ext.source.tile.Config.url` = `Quell-URL mit Platzhaltern {x}, {y} und {z}`
- `gws.ext.source.wms.Config` = `WMS Quelle`
- `gws.ext.source.wms.Config.capsCacheMaxAge` = `Maximales Cache-Alter für Capabilities-Dokumente`
- `gws.ext.source.wms.Config.crs` = `KBS für diese Quelle`
- `gws.ext.source.wms.Config.extent` = `Layerausmasse`
- `gws.ext.source.wms.Config.maxRequests` = `Maximal gleichzeitige Anforderungen an diese Quelle`
- `gws.ext.source.wms.Config.options` = `Zusatzoptionen`
- `gws.ext.source.wms.Config.params` = `Abfragezeichenfolgeparameter`
- `gws.ext.source.wms.Config.type` = `Objekttyp`
- `gws.ext.source.wms.Config.url` = `Service-URL`
- `gws.ext.source.wmts.Config` = `WMTS Quelle`
- `gws.ext.source.wmts.Config.capsCacheMaxAge` = `Maximales Cache-Alter für Capabilities-Dokumente`
- `gws.ext.source.wmts.Config.crs` = `KBS für diese Quelle`
- `gws.ext.source.wmts.Config.extent` = `Layerausmasse`
- `gws.ext.source.wmts.Config.format` = `Bildformat`
- `gws.ext.source.wmts.Config.layer` = `WMTS Layername`
- `gws.ext.source.wmts.Config.options` = `Zusatzoptionen`
- `gws.ext.source.wmts.Config.style` = `Bildstil`
- `gws.ext.source.wmts.Config.tileMatrixSet` = `WMTS Tile Matrixsatz`
- `gws.ext.source.wmts.Config.type` = `Objekttyp`
- `gws.ext.source.wmts.Config.url` = `Service-URL`
- `gws.ext.template.html.Config` = `HTML Vorlage`
- `gws.ext.template.html.Config.dataModel` = `Vom Benutzer editierbare Vorlagenattribute`
- `gws.ext.template.html.Config.mapHeight` = `Kartenhöhe`
- `gws.ext.template.html.Config.mapWidth` = `Kartenbreite`
- `gws.ext.template.html.Config.margins` = `Seitenraender`
- `gws.ext.template.html.Config.pageHeight` = `Seitenhoehe`
- `gws.ext.template.html.Config.pageWidth` = `Seitenbreite`
- `gws.ext.template.html.Config.path` = `Pfad zu einer Vorlagendatei`
- `gws.ext.template.html.Config.qualityLevels` = `Liste der von der Vorlage unterstuetzten Qualitaetsstufen`
- `gws.ext.template.html.Config.text` = `Vorlageninhalt`
- `gws.ext.template.html.Config.title` = `Vorlagentitel`
- `gws.ext.template.html.Config.type` = `Vorlagentyp`
- `gws.ext.template.html.Config.unit` = `Einheiten für Karten- . Seitengroessen`
- `gws.ext.template.qgis.Config` = `QGIS Drucktemplate`
- `gws.ext.template.qgis.Config.dataModel` = `Vom Benutzer editierbare Vorlagenattribute`
- `gws.ext.template.qgis.Config.qualityLevels` = `Liste der von der Vorlage unterstützten Qualitätsstufen`
- `gws.ext.template.qgis.Config.text` = `Vorlageninhalt`
- `gws.ext.template.qgis.Config.title` = `Vorlagentitel`
- `gws.ext.template.qgis.Config.type` = `Vorlagentyp`
- `gws.gis.BaseConfig.crs` = `KBS für diese Quelle`
- `gws.gis.BaseConfig.extent` = `Layerausmasse`
- `gws.gis.BaseConfig.type` = `Objekttyp`
- `gws.lib.zoom.Config` = `Zoom- und Aufloesungskonfiguration`
- `gws.lib.zoom.Config.initResolution` = `Ausgangsaufloesung`
- `gws.lib.zoom.Config.initScale` = `Ausgangsmassstab`
- `gws.lib.zoom.Config.maxResolution` = `maximale Aufloesung`
- `gws.lib.zoom.Config.maxScale` = `maximaler Massstab`
- `gws.lib.zoom.Config.minResolution` = `Minimale Aufloesung`
- `gws.lib.zoom.Config.minScale` = `Minimaler Massstab`
- `gws.lib.zoom.Config.resolutions` = `Erlaubte Aufloesungen`
- `gws.lib.zoom.Config.scales` = `Erlaubte Massstaebe`
- `gws.types.Access` = `Zugriffsrechte definition for authorization roles`
- `gws.types.Access.mode` = `Zugriffsmodus`
- `gws.types.Access.role` = `Liste der Rollen, fuer die diese Regel gilt`
- `gws.types.Access.type` = `Zugriffstyp (deny oder allow)`
- `gws.types.AttributeConfig` = `Attributkonfiguration`
- `gws.types.AttributeConfig.name` = `Interner Name`
- `gws.types.AttributeConfig.title` = `Titel`
- `gws.types.AttributeConfig.type` = `Typ`
- `gws.types.CacheConfig` = `Karten Cache-Konfiguration`
- `gws.types.CacheConfig.enabled` = `Cache ist aktiviert`
- `gws.types.CacheConfig.maxAge` = `Cache max. Alter`
- `gws.types.CacheConfig.maxLevel` = `max. Zoomstufe zum Cachen`
- `gws.types.CacheConfig.options` = `zusätzliche MapProxy Cache-Optionen`
- `gws.types.ConfigWithAccess.access` = `Zugriffsrechte`
- `gws.types.ConfigWithAccess.type` = `Objekttyp`
- `gws.types.DocumentRootConfig` = `Asset-Basisverzeichniskonfiguration`
- `gws.types.DocumentRootConfig.allowMime` = `erlaubte Mime-Typen`
- `gws.types.DocumentRootConfig.denyMime` = `nicht erlaubte Mime-Typen (von Standardliste)`
- `gws.types.DocumentRootConfig.dir` = `Verzeichnispfad`
- `gws.types.FormatConfig` = `Objektformat`
- `gws.types.FormatConfig.category` = `Objektkategorie`
- `gws.types.FormatConfig.description` = `Vorlage fuer Objektbeschreibung`
- `gws.types.FormatConfig.label` = `Objektbeschriftung in der Karte`
- `gws.types.FormatConfig.model` = `Attributtransformationsregeln`
- `gws.types.FormatConfig.teaser` = `Vorlage fuer Objekt-Teaser (Kurzbeschreibungen)`
- `gws.types.FormatConfig.title` = `Objekttitel`
- `gws.types.GridConfig` = `Rasterkonfiguration fuer gekachelte oder zwischengespeicherte Kartendaten`
- `gws.types.GridConfig.options` = `zusätzliche MapProxy-Rasteroptionen`
- `gws.types.GridConfig.origin` = `Position der ersten Kachel (nw oder sw)`
- `gws.types.GridConfig.reqBuffer` = `Pixelpuffer`
- `gws.types.GridConfig.reqSize` = `Anzahl der Meta-tiles zum Abrufen`
- `gws.types.GridConfig.tileSize` = `Kachelgroesse`
- `gws.types.Record` = `Objektmetadaten`
- `gws.types.Record.abstract` = `Beschreibung der Objektzusammenfassung`
- `gws.types.Record.attribution` = `Attribut (Copyright) Angabe`
- `gws.types.Record.image` = `Bild (Logo) URL`
- `gws.types.Record.images` = `weitere Bilder`
- `gws.types.Record.keywords` = `Schluesselworte`
- `gws.types.Record.name` = `Interner Objektname`
- `gws.types.Record.title` = `Objekttitel`
- `gws.types.Record.url` = `Objekt Metadaten-URL`
- `gws.types.SqlTableConfig` = `SQL Datenbanktabelle`
- `gws.types.SqlTableConfig.geometryColumn` = `Name der Geometriespalte`
- `gws.types.SqlTableConfig.keyColumn` = `Name der Primary-Key-Spalte`
- `gws.types.SqlTableConfig.name` = `Tabellenname`
- `gws.types.SqlTableConfig.searchColumn` = `Spalte, nach der gesucht werden soll`
- `gws.types.StyleProps` = `Objektstil`
- `gws.types.StyleProps.content` = `CSS Regeln`
- `gws.types.StyleProps.type` = `Stiltyp (."css.")`
- `gws.types.StyleProps.value` = `reiner Stilinhalt`
- `gws.types.TemplateConfig.dataModel` = `Vom Benutzer editierbare Vorlagenattribute`
- `gws.types.TemplateConfig.path` = `Pfad zu einer Vorlagendatei`
- `gws.types.TemplateConfig.qualityLevels` = `Liste der von der Vorlage unterstuetzten Qualitaetsstufen`
- `gws.types.TemplateConfig.text` = `Vorlageninhalt`
- `gws.types.TemplateConfig.title` = `Vorlagentitel`
- `gws.types.TemplateConfig.type` = `Vorlagentyp`
- `gws.types.TemplateQualityLevel` = `benannte Qualitaetsstufe fuer Vorlagen`
- `gws.types.TemplateQualityLevel.dpi` = `DPI-Wert`
- `gws.types.TemplateQualityLevel.name` = `Levelname`
- `gws.types.WithType.type` = `Objekttyp`
- `gws.types.crsref` = `KBS code wie ."EPSG.3857."`
- `gws.types.dirpath` = `gueltiger lesbarer Verzeichnispfad auf dem Server`
- `gws.types.duration` = `Zeichenfolge wie '1w 2d 3h 4m 5s' oder eine ganze Anzahl Sekunden`
- `gws.types.filepath` = `gueltiger lesbarer Dateipfad auf dem Server`
- `gws.types.formatstr` = `Text mit {attribute} Platzhaltern`
- `gws.types.regex` = `regulaerer Ausdruck, wie er in Python verwendet wird`
- `gws.types.url` = `http oder https URL`
