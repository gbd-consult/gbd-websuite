# Schnellstart :/admin-de/schnellstart

<!--
    Schreibstil:
    Aussagen über die GBD WebSuite als kurze, bündige Fakten.
        z.B.  Die GBD WebSuite wird als Container ausgeliefert. 
    Handlungsaufforderungen als imperativ, förmliche Anrede
        z.B.  Erstellen Sie eine Datei mit dem Namen `docker-compose.yml`:
    
    Zielsetzung:
    - So kurz und bündig wie möglich.
    - Verweis auf Dokumentation benötigter Fremdsoftware.
    - Führen des Benutzers durch minimal-Beispiel zum Starten der WebSuite und betrachten des mitgelieferten Demo-Projektes.
    - Optional: Vorbereitung auf "Guide: Einfaches Projekt"
-->

%include snippets/release-info.md

## Docker
Die GBD WebSuite wird als Container ausgeliefert. 
Um einen Container zu starten benötigen Sie eine Container Runtime wie z.B. 
[Docker](https://docker.com). 
In diesem Beispiel verwenden wir Docker mit dem 
Compose Plugin: [Docker Engine Installationsanleitung](https://docs.docker.com/engine/install/#supported-platforms).

## docker-compose.yml
Erstellen Sie eine Datei mit dem Namen `docker-compose.yml` und dem folgenden 
Inhalt in einem beliebigen Verzeichnis:

```yaml title="docker-compose.yml"
services:
    gws:
        image: gbdconsult/gws-amd64:8.2
        container_name: gws
        ports:
            - 3333:80
```


## Download & Starten
Weisen Sie Docker an das benötigte Image herunterzuladen  

    docker compose -f docker-compose.yml pull

und den Container zu starten: 

    docker compose -f docker-compose.yml up

%info

Sollten Sie eine Fehlermeldung mit dem Inhalt `The container name "/gws" is already in use` erhalten, haben Sie die WebSuite vermutlich schon einmal gestartet. In dem Fall können Sie den veralteten Container mittels dem folgenden Befehl löschen:

    docker rm gws

%end

Abhängig von Ihrem Betriebssystem und User müssen Sie dies möglicherweise als 
Administrator (auf Ubuntu z.B. mit `sudo`) tun, oder Ihrem Benutzer die benötigten
Rechte geben (auf Ubuntu `sudo adduser <myuser> docker`).

## Betrachten im Browser
Sobald der Download und der Startvorgang abgeschlossen sind, können Sie in ihrem 
Browser auf http://localhost:3333 das Testprojekt der GBD WebSuite sehen.


## Beenden
Um die GBD WebSuite wieder zu beenden drücken Sie in dem Terminal in dem Sie 
aktuell die Logausgabe sehen STRG+C, oder schließen dieses.


## Wo mache ich weiter?

Um basierend auf diesem Schnellstart anzufangen mit der Konfiguration der GBD 
WebSuite zu experimentieren, fahren Sie wie folgt fort:

1. *(falls noch nicht geschehen)* Erstellen Sie ein Verzeichnis in dem alle Konfigurationsdateien für die GBD WebSuite abgelegt werden.
2. Verschieben Sie die `docker-compose.yml` in dieses Verzeichnis.
3. Erstellen Sie ebenfalls in diesem Verzeichnis die Unterordner `data` und `gws-var`.
4. Ersetzen Sie den Inhalt der `docker-compose.yml` durch:

```yaml title="MYGWSDIRECTORY/docker-compose.yml"
services:
    gws:
        image: gbdconsult/gws-amd64:8.2
        container_name: gws
        volumes:
            - ./data:/data
            - ./gws-var:/gws-var
        ports:
            - 3333:80
        tmpfs:
            - /tmp
    qgis:
        image: gbdconsult/gbd-qgis-server-amd64:3.34
        container_name: qgis
        volumes:
            - ./data:/data
            - ./gws-var:/gws-var
        tmpfs:
            - /tmp
```
5. Entfernen Sie den im ersten Anlauf gestarteten Container mittels `docker rm gws`
6. Wechseln Sie im Terminal/Eingabeaufforderung in das erstellte Konfigurationsverzeichnis.

Die GBD WebSuite wird dann zunächst nicht mehr funktionieren, da Sie das 
Konfigurationsverzeichnis auf das so eben erstellte `data` Verzeichnis 
umgeleitet haben und dieses aktuell noch leer ist. Wie Sie dieses Verzeichnis 
mit einer initiellen Konfiguration füllen können wird hier beschrieben: 

[**Guide: Einfaches Projekt**](/admin-de/einfaches-projekt)

----

Wie Sie die GBD WebSuite auf einem Server installieren können, so dass diese 
Installation auch als Produktivumgebung genutzt werden kann wird hier 
beschrieben: [Guides/Installation](/admin-de/guides/installation)
