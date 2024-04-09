# Schnellstart :/admin-de/guides/schnellstart

%include snippets/release-info.md

## Docker
Die GBD WebSuite wird als Container ausgeliefert. 
Um einen Container zu starten benötigen Sie eine Container Runtime wie z.B. 
[Docker](https://docker.com). 
In diesem Beispiel verwenden wir Docker mit dem 
Compose Plugin: [Docker Engine Installationsanleitung](https://docs.docker.com/engine/install/).


## docker-compose.yml
Erstellen Sie eine Datei mit dem Namen `docker-compose.yml` und dem folgenden 
Inhalt in einem beliebigen Verzeichnis:

```yaml
version: '3'

services:
    gws:
        image: gbdconsult/gws-amd64:8.0
        container_name: gws
        ports:
            - 3333:80
```


## Download & Starten
Weisen Sie Docker an das benötigte Image herunterzuladen (ca. 500MB) und den 
Container zu starten: 

    docker compose -f docker-compose.yml up


## Betrachten im Browser
Sobald der Download und der Startvorgang abgeschlossen sind, können Sie in ihrem 
Browser auf http://localhost:3333 das Testprojekt der GBD WebSuite sehen.


## Beenden
Um die GBD WebSuite wieder zu beenden drücken Sie in dem Terminal in dem Sie 
aktuell die Logausgabe sehen STRG+C, oder schließen dieses.


## Wo mache ich weiter?

Um basierend auf diesem Schnellstart anzufangen mit der Konfiguration der GBD 
WebSuite zu experimentieren, erweitern Sie die `docker-compose.yml` wie folgt:

```yaml
version: '3'

services:
    gws:
        image: gbdconsult/gws-amd64:8.0
        container_name: gws
        volumes:
            - MYGWSDIRECTORY/data:/data
            - MYGWSDIRECTORY/gws-var:/gws-var
        ports:
            - 3333:80
        tmpfs:
            - /tmp
    qgis:
        image: gbdconsult/qgis-amd64:8.0
        container_name: qgis
        volumes:
            - MYGWSDIRECTORY/data:/data
            - MYGWSDIRECTORY/gws-var:/gws-var
        tmpfs:
            - /tmp
```

Passen Sie die Verzeichnis-Pfade unter `volumes:` für beide Container ihrem 
System entsprechend an. Ersetzen Sie dafür alle Vorkommen von `MYGWSDIRECTORY` 
durch einen für Ihr System gültigen Pfad.

Unter Windows könnte das z.B. `C:/Users/<myusername>/Desktop/gwstest` sein, und 
unter Linux `/home/<myusername>/gwstest`.

Stellen Sie sicher, dass die Verzeichnisse `data` und `gws-var` innerhalb des 
gwstest Verzeichnisses existieren.

Die GBD WebSuite wird dann zunächst nicht mehr funktionieren, da Sie das 
Konfigurationsverzeichnis auf das so eben erstellte `gwstest/data` Verzeichnis 
umgeleitet haben und dieses aktuell noch leer ist. Wie Sie dieses Verzeichnis 
mit einer initiellen Konfiguration füllen können wird hier beschrieben: TODO LINK 

----

Wie Sie die GBD WebSuite auf einem Server installieren können, so dass diese 
Installation auch als Produktivumgebung genutzt werden kann wird hier 
beschrieben: [Guides/Installation](/admin-de/guides/installation)