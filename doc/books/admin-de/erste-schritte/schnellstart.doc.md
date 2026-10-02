# Schnellstart :/admin-de/erste-schritte/schnellstart

%info
Arbeiten Sie möglichst mit dem aktuellen stabilen Release. Eine Übersicht der Releases finden Sie unter https://gbd-websuite.de
%end

Die GBD WebSuite wird als Docker-Container ausgeliefert. Für dieses Beispiel genügt eine Container-Runtime wie [Docker](https://docker.com) mit dem Compose-Plugin (siehe [Installationsanleitung](https://docs.docker.com/engine/install/)).

## Container starten

Erstellen Sie in einem beliebigen Verzeichnis eine Datei `docker-compose.yml`:

```yaml title="docker-compose.yml"
services:
    gws:
        image: gbdconsult/gws-amd64:8.4
        container_name: gws
        ports:
            - 3333:80
```

Image laden und Container starten:

    docker compose pull
    docker compose up

%info
Erscheint die Meldung `The container name "/gws" is already in use`, läuft bereits ein alter Container. Entfernen Sie ihn mit `docker rm gws`.
%end

Je nach Betriebssystem sind dafür Administratorrechte nötig – unter Ubuntu z. B. mit `sudo`, oder indem Sie Ihren Benutzer mit `sudo adduser <benutzer> docker` berechtigen.

## Demo-Projekt ansehen

Sobald der Startvorgang abgeschlossen ist, erreichen Sie das mitgelieferte Demo-Projekt im Browser unter http://localhost:3333.

Das Demo-Projekt ist Teil des Images und benötigt keine eigene Konfiguration. Es zeigt eine Karte mit mehreren Layern und die wichtigsten Client-Elemente – Ebenenbaum, Suche und Werkzeugleiste – und eignet sich daher gut, um zu prüfen, ob Container und Netzwerk richtig eingerichtet sind.

Zum Beenden drücken Sie im Terminal STRG+C.

## Eigene Konfiguration vorbereiten

Bis hierher läuft die WebSuite mit der eingebauten Demo-Konfiguration. Für eine eigene Konfiguration binden Sie zwei Verzeichnisse als Volumes ein: `data` für Konfiguration und Daten, `gws-var` für veränderliche Laufzeitdaten. Wenn Sie QGIS-Projekte nutzen möchten, kommt ein QGIS-Server-Container hinzu.

```yaml title="docker-compose.yml"
services:
    gws:
        image: gbdconsult/gws-amd64:8.4
        container_name: gws
        volumes:
            - ./data:/data
            - ./gws-var:/gws-var
        ports:
            - 3333:80
        tmpfs:
            - /tmp
    qgis:
        image: gbdconsult/gbd-qgis-server-amd64:3.44
        container_name: qgis
        volumes:
            - ./data:/data
            - ./gws-var:/gws-var
        tmpfs:
            - /tmp
```

Legen Sie die Verzeichnisse `data` und `gws-var` neben der `docker-compose.yml` an und entfernen Sie den zuvor gestarteten Container mit `docker rm gws`. Da `data` noch leer ist, findet die WebSuite zunächst keine Konfiguration – wie Sie diese anlegen, zeigen die folgenden Kapitel.

## Wie geht es weiter?

- [](/admin-de/erste-schritte/installation) – Betrieb als Produktivumgebung.
- [](/admin-de/erste-schritte/konfigurationsgrundlagen) – Syntax und Aufbau der Konfiguration.
- [](/admin-de/erste-schritte/einfaches-projekt) – erstes Projekt mit Karte anlegen.
