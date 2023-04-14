# Installation :/admin-de/intro/install

## docker-compose.yml

```yaml
version: '3'
services:
    gws:
        image: gbdconsult/gws-amd64:8.0.0
        container_name: gws
        ports:
            - "80:80"
            - "443:443"
        volumes:
            - "~/gws-welcome/data:/data:ro"
            - "~/gws-var/print:/gws-var/print"
            - "~/gws-var:/gws-var"
      #     - "~/gws-websuite/app:/gws-app:ro"
      # extra_hosts:
      # 	- "somehost:162.242.195.82"
      # 	- "otherhost:50.31.209.229"
      # environment:
      #     GWS_UID=1000 # default 1000
      #     GWS_GID=1000 # default 1000
      #     GWS_CONFIG=/data/config.json # default /data/config.json
    qgis:
        image: gbdconsult/qgis-3.28-amd64:0
        container_name: qgis
        ports:
            - "80"
        volumes:
            - "~/gws-welcome/data:/data:ro"
            - "~/gws-var/print:/gws-var/print:ro"
        environment:
            QGIS_DEBUG=0
            QGIS_WORKERS=1
```

## Voraussetzungen

### Verzeichnisse

Innerhalb des Containers existieren die folgenden relevanten Verzeichnisse:

`/data`: Enhält Konfiguration und Daten für die Anwendung (z.B. .json, .qgs, .geotiff). Der Container muss in diesem Verzeichnis nicht schreiben. Wird ein Verzeichnis auf dem Host in den Container gemounted, sollte dies readonly (`:ro`) geschehen. Es ist bei Bedarf möglich weitere Datenverzeichnisse an beliebige Orte in den Container zu mounten. Diese müssen dann in der Konfiguration explizit referenziert werden.

`/gws-var`: Enhält Cache und Sitzungsdaten. Die GBD WebSuite muss in dieses Verzeichnis schreiben können. Das Verzeichnis muss entweder vom Hostsystem in den Container gemounted werden, oder als persistentes Volume existieren damit Daten nicht zwischen Updates/Neustarts verlorengehen.

`/gws-var/print`: Muss sowohl vom qgis- als auch vom gws-Container schreibbar sein, um einen Datenaustausch zwischen qgis-server und GBD WebSuite zu ermöglichen. Wird für Druckfunktionalitäten verwendet. Muss nicht persistieren.

`/gws-app`: Enhält den Quellcode der GBD WebSuite Applikation. Nur für Entwickler interessant. In einer normalen Installation die das fertige Image von hub.docker.com verwendet ist der Quellcode bereits im Container hinterlegt. Ein Update geschieht im Normalfall durch Aktualisierung des gesamten Containers.

`TODO: - ein temporäres Verzeichnis *tmp*. Normalerweise würde man es als `tmpfs` bezeichnen.`

### Benutzerkonten

Innerhalb des Docker Containers hat das Benutzerkonto mit dem die GBD WebSuite ausgeführt *standardmäßig* die User- und Gruppen ID `1000`. Um ein problemloses Lesen und Schreiben auf aus dem Hostsystem in den Container gemountete Verzeichnisse zu gewährleisten muss das Hostsystem-Benutzerkonto mit der gleichen ID die entsprechenden Rechte auf die relevanten Verzeichnisse des Hostsystems haben.

Sollte es in Ihrer Infrastruktur nicht möglich sein dem Benutzer mit der ID 1000 die nötigen Rechte zu geben können Sie die innerhalb des Containers verwendete User- und Gruppen ID anpassen in dem Sie die folgenden Umgebungsvariablen beim Start des Containers übergeben: `GWS_UID`, `GWS_GID`.

### Ports

Die GBD WebSuite antwortet auf HTTP und HTTPS Anfragen auf den Standardports 80 und 443. Haben Sie durch Hinterlegen von Zertifikaten in der Konfiguration HTTPS aktiviert, so wird auf 80/HTTP ein *permanenter* Redirect auf 443/HTTPS hinterlegt.

### Konfiguration

Der Einstiegspunkt für die Konfiguration ist per Default `/data/config.json`. Alle weiteres Konfigurationsdateien werden von dieser Datei eingebunden.
Um eine andere Datei als Einstiegspunkt zu verwenden setzen Sie die Umgebungsvariable `GWS_CONFIG`.

^SEE Die Konfigurationsformate sind unter ^config/intro beschrieben.

### Externe Adressen

Wenn Ihr GBD WebSuite Container externe Verbindungen benötigt (höchstwahrscheinlich zu Datenbankservern), müssen Sie die Adressen zu diesen Hosts explizit hinterlegen.
In der `docker-compose.yml` ist dies unter dem Punkt `extra_hosts` möglich. Siehe Beispiel oben.



### Einstiegspunkt (alt?)

Die GBD WebSuite hat einen einzigen Einstiegspunkt, ein Shell-Skript namens `gws`. Um den Server zu starten oder zu stoppen, nutzen Sie die folgenden Befehle:

    gws server start
    gws server stop

## Ausführen mit `gws`-Shell-Skript (alt?)

Folgende Optionen müssen Sie in Ihrem `gws`-Shell-Skript anpassen:

- ein oder mehrere *Daten*-Verzeichnisse einbinden
- *var* und *tmp* Verzeichnisse einbinden
- Port-Mappings
- Konfigurationspfad
- externe Hosts

Hier ist eine Skript-Vorlage, die die GBD WebSuite mittels `docker run` startet. Sie können diese Vorlage nach Ihren Bedürfnissen anpassen.

@quote
    #!/usr/bin/env bash

    RELEASE=6.1
    CONTAINER=gws-container
    DOCKER='sudo docker'
    IMAGE=gbdconsult/gws-server:$RELEASE
    LOG_DRIVER=syslog
    DATA_DIR=/your/gws/data/directory
    VAR_DIR=/your/gws/var/directory
    CONFIG_PATH=/data/config.cx

    STARTOPTS=(
        --name $CONTAINER
        --mount type=bind,src=$DATA_DIR,dst=/data,readonly
        --mount type=bind,src=$VAR_DIR,dst=/gws-var
        --mount type=tmpfs,dst=/tmp,tmpfs-mode=1777
        --publish 0.0.0.0:80:80
        --publish 0.0.0.0:443:443
        --log-driver $LOG_DRIVER
        --log-opt tag=GWS
        --env GWS_CONFIG=$CONFIG_PATH
    )

    start_server() {
        $DOCKER run ${STARTOPTS[*]} --detach $IMAGE gws server start
    }

    stop() {
        $DOCKER exec $CONTAINER gws server stop
        $DOCKER stop $CONTAINER
        $DOCKER rm --force $CONTAINER
    }

    case "$1_$2" in
        server_start)   start_server ;;
        server_stop)    stop  ;;
        server_restart) stop; start_server ;;
    esac
@end

Sobald Sie dieses Skript als z.B. `gws` in Ihren Pfad abspeichern, können Sie die WebSuite mit diesen Befehlen steuern:

    gws server start
    gws server stop
    gws server restart

## Aktuellen Quellcode anbinden (alt?)

Da die GBD WebSuite aktiv entwickelt wird, kann es vorkommen, dass eine in dem Docker-Image enthaltene Version von unserem Quellcode veraltet ist. Sie können aber das Image mit der aktuellen Version ausführen indem Sie das Quellcodeverzeichnis unter `gws-app` mounten.

Laden Sie zuerst unser Paket von Ihrer Version herunter (in diesem Fall, Version 6.1):

    curl -O http://gws-files.gbd-consult.de/gws-6.1.tar.gz

entpacken Sie das Paket:

    tar xvzf gws-6.1.tar.gz

und mounten Sie den `gws-server/app` Unterordner als `gws-app`:

    --mount type=bind,src=<absoluter Pfad>/gws-server/app,dst=/gws-app,readonly

## Host-Installation (alt?)

Wir haben auch ein Skript, das die WebSuite direkt auf Ihrem System installiert, ohne docker. Das Skript finden Sie in unserem Github unter https://github.com/gbd-consult/gbd-websuite/blob/master/install/install.sh

.. caution:: Diese Entwicklung ist experimentell, nicht auf Produktionsserver probieren!
