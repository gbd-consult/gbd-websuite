# Administrator Handbuch :/admin-de

Dokumentation für GBD WebSuite Server Administratoren (Deutsch).

## Einführung :intro

### Was ist die GBD WebSuite

Bei der GBD WebSuite handelt es sich um einen Anwendungs- und Webserver, welcher den Schwerpunkt auf die Geodatenverarbeitung legt.

Die GBD WebSuite als Webserver:

- kann statische- und Template-Inhalte bedienen
- unterstützt Multi-Site-Konfigurationen, URL-Routing und Rewriting
- unterstützt verschiedene Berechtigungsmechanismen (Dateisystem, Datenbank, LDAP) und feinkörnige Berechtigungen

Die GBD WebSuite als Geo-Server:

- kombiniert verschiedene Quellen (WMS, Kachelserver, Datenbanken) zu einer einheitlichen Karte
- hat direkte Unterstützung für QGIS-Projekte
- Caches, Reprojekte und Skalierung von Rasterdaten nach Bedarf kann Vektordaten verarbeiten und rendern (PostGIS, Shapes, JSON)
- bietet OGC-konforme Dienste an (WMS, WMTS, WFS)

Die GBD WebSuite als Anwendungsserver:

- bietet einen Rahmen für domänenspezifische Erweiterungen
- hat eine modulare Architektur für einfache Integration

Die GBD WebSuite ist eine Docker-Anwendung, die die folgenden Komponenten beinhaltet:

- `NGINX <https://www.nginx.com/>`_ Webserver, der für statische Inhalte sowie URL-Rewriting zuständig ist
- `uWSGI <https://github.com/unbit/uwsgi>`_ Application Server, der dynamische Anfragen bearbeitet
- Python Komponente ("Aktionen"), die für Datenabfragen und Verarbeitung zuständig sind
- `QGIS <https://qgis.org>`_ Server, zum Rendern von QGIS Projekten
- `MapProxy <https://mapproxy.org/>`_ Server, zum Cachen von Kartenbildern

Die GBD WebSuite basiert vollständig auf Free Open Source Software, und ist eine frei zugängliche Software (Apache License 2.0).

### Wie ist dieses Buch aufgebaut

Wenn Sie die GBD WebSuite zum ersten Mal nutzen, starten Sie mit dem Kapitel ^quickstart, welches den ersten Start des Servers und Ihres ersten Projektes beschreibt. Im Kapitel ^install wird die Installation der GBD WebSuite ausführlicher beschrieben.  In dem Kapitel ^concepts werden dann die Grundkonzepte und Funktionen der GBD WebSuite vorgestellt. Im Kapitel ^config/index finden Sie detaillierte Anweisungen zur Konfiguration. In dem Kapitel ^ref/index finden Sie eine Auflistung aler Konfigurationsoptionen sowie aller Kommandozeilen Tools.

### Schnellstart :quickstart

Hiermit starten Sie den GBD WebSuite Server zum ersten Mal und richten Ihr erstes Projekt ein.

**Schritt 1**. Stellen Sie sicher, dass `docker <https://www.docker.com>`_ installiert ist und funktioniert.

**Schritt 2**. Laden Sie das GBD WebSuite Server Image herunter und testen Sie es

    docker run -it -p 3333:80 --name my-gws-container gbdconsult/gws-server:latest

Dies führt den GBD WebSuite Server auf dem Port `3333` unter dem Namen `my-gws-container` aus (zögern Sie nicht, einen anderen Namen und/oder Port zu verwenden). Bei einigen Setups benötigen Sie möglicherweise Root-Rechte (`sudo`), um dies ausführen zu können.

Wenn alles in Ordnung ist, sollten Sie das Server-Log auf Ihrem Terminal sehen. Gehen Sie in Ihrem Browser auf http://localhost:3333. Es wird die Server-Startseite und unser Beispielprojekt gezeigt.

Stoppen Sie nun den Server mit Control-C und entfernen Sie den Container

    docker rm my-gws-container

**Schritt 3**. Erstellen Sie ein Verzeichnis `data` irgendwo auf Ihrer Festplatte (z. B. `/var/work/data`). Laden Sie die folgenden Daten herunter und speichern Sie die ab in dieses Verzeichnis:

- https://github.com/gbd-consult/gbd-websuite/tree/master/doc/examples/quickstart/config.json
- https://github.com/gbd-consult/gbd-websuite/tree/master/doc/examples/quickstart/project.html

**Schritt 4**. Starten Sie den Container erneut und geben Sie ihm einen Pfad zu Ihrer neu erstellten Konfiguration

    docker run -it -p 3333:80 --name my-gws-container --mount type=bind,src=/var/work/data,dst=/data gbdconsult/gws-server:latest

Navigieren Sie zu http://localhost:3333/hello. Sie sollten die OpenStreetMap von Düsseldorf sehen, der Geburtsort der GBD WebSuite.

### :concepts
### :install

## Grundlagen der Konfiguration

In diesem Kapitel finden Sie eine detaillierte Beschreibung der GBD WebSuite Konfiguration. 

Die GBD WebSuite Konfiguration besteht aus Applikations-Konfigurationsdatein (^app) und optional mehrere Projekt-Konfigurationsdateien (^project). Die Struktur der Konfigurationsobjekte ist in ^../ref/config vollständig beschrieben.

### Konfigurationsformate

Die Konfirgurationen können in verschiedenen Sprachen geschrieben werden, nämlich JSON, YAML, SLON und Python. Sie können die Sprachen auch frei mischen, z.B. App-Konfig in Python und Projekt-Konfig in YAML.

#### json

Bei JSON (https://www.json.org) handelt es sich um ein gängiges Konfigurations- und Datenaustauschformat. In dieser Dokumentation verwenden wir JSON für Code-Snippets und Beispiele. Dies ist auch unser Defaultformat: falls Sie keinen expliziten Konfigurationspfad mittels `GWS_CONFIG` bestimmen, wird eine JSON Datei `config.json` im "Data"-Verzeichnis geladen. JSON Konfigdateien müssen mit der Endung `.json` abgespeichert werden.

#### yaml

YAML (https://yaml.org) ist eine Alternative zu JSON, die einfacher zu schreiben und lesen ist. Sie können Ihre Konfiguration in YAML schreiben, mit der gleichen Struktur wie JSON. YAML Konfigdateien müssen mit der Endung `.yaml` abgespeichert werden.

#### slon

SLON (https://github.com/gebrkn/slon) ist  eine Alternative zu JSON, die die Strukturen in einer weiter vereinfachten Form darstellt. Bei diesem Format können Sie auch alle Befehle der Templating-Sprache verwenden (wie z.B. `@include` oder `@if`). Konfigdateien im SLON Format müssen eine Erweiterung `.cx` haben.

^SEE Mehr über Templating-Sprache lesen Sie unter ^template.

#### python

Komplexe, sich wiederholende oder hochdynamische Konfigurationen können auch direkt in def Programmiersprache Python geschrieben werden. Die Python-Konfigurationsdatei muss eine Funktion `config()` enthalten, die einen Python `dict` zurückgibt. Beachten Sie, dass Ihr Konfigurationsmodul innerhalb des Containers ausgeführt wird und daher mit Python 3.6 kompatibel sein muss.

### Struktur der Konfiguration

Auf der obersten Ebene, ist die Konfiguration eine Schlüssel-Wert Struktur (*dict*), die Zeichenketten als Schlüssel und entweder die "primitiven" Werte (wie z.B. eine Zahl oder Zeichenkette) oder weitere Schlüssel-Wert Strukturen bzw. Listen (*arrays*) von Werten enthält.

Einige Schlüssel-Wert Strukturen haben eine grundlegende Eigenschaft Typ (`type`), der angibt, zu welchem Typ die gegebene Struktur gehört. Diese Eigenschaft ist stets anzugeben.

Eine weitere grundlegende Eigenschaft, def Identifikator (`uid`), ist dagegen Optional und ist nur dann anzugeben, wenn Sie auf die gegebene Struktur an weiteren Stellen der Konfiguration verweisen möchten. In anderen Fällen wird die `uid` aus dem Objekt-Titel bzw. Typ automatisch generiert.

### Laden der Konfiguration

Die Konfiguration beginnt mit der App-Konfigurationsdatei (`GWS_CONFIG` bzw `data/config.json`). Sobald diese Datei erfolgreich gelesen ist, werden die Projekte geladen. Die Projekte werden mit drei Optionen in der App-Konfig bestimmt:

- `projects` (eine Liste von Projekten). Mit dieser Option werden Ihre Projekte direkt in der App-Konfig eingebunden.

- `projectPaths` (eine Liste von Dateinamen). Mit dieser Option werden Projekte aus angegebenen Dateien gelesen, wobei jede Datei ein Projekt oder eine Liste von Projekten enthält

- `projectDirs` (eine Liste von Ordnernamen). Mit dieser Option liest das System aus angegebenen Verzeichnissen alle Dateien die mit `.config.json`, `.config.yaml`, `.config.cx` oder `.config.py` enden und diese als Projekte konfiguriert.

Diese Optionen können miteinander auch frei kombiniert werden.

### Monitoring

GWS Server enthält ein *Monitor* Modul, der das Dateisystem überwacht, die Änderungen in Ihren Projekten und Konfigurationen überprüft und ggf. einen Hot-Reload des Servers durchführt. Sie können Intervalle für diese Prüfungen konfigurieren, es wird empfohlen, das Monitorintervall auf mindestens 30 Sekunden einzustellen, da Dateisystemprüfungen ressourcenintensiv sind.

^SEE Sie können Monitoring unter ^server konfigurieren.

## Konfiguration A-Z :config

Application und System Konfiguration:

%toc
/admin-de/config/app
/admin-de/config/auth
/admin-de/config/client
/admin-de/config/db
/admin-de/config/intl
/admin-de/config/metadata
/admin-de/config/server
/admin-de/config/web
%end

Konfiguration der Karten, Layer und Suchoptionen.

%toc
/admin-de/config/map
/admin-de/config/layer
/admin-de/config/feature
/admin-de/plugin/qgis
/admin-de/config/cache
/admin-de/config/search
/admin-de/config/metadata
%end

Options für die visuelle Präsentation von Karten und Sachdaten.

%toc
/admin-de/config/template
/admin-de/config/style
/admin-de/config/print
/admin-de/config/csv
%end

Optionen in Bezug auf Editieren von geografischen Objekten und Sachdaten.

%toc
/admin-de/plugin/edit
/admin-de/plugin/tabedit
%end

### :/admin-de/config/*

## Plugins :plugin

Die GBD WebSuite unterstützt viele Plugins.

%toc
/admin-de/plugin/*
%end

### :/admin-de/plugin/*

## :/admin-de/reference
