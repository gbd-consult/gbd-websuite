# Installation :/admin-de/erste-schritte/installation

Dieses Kapitel behandelt die Installation der GBD WebSuite für den produktiven Betrieb – nicht die des Betriebssystems oder von Docker.

Gegenüber dem [](/admin-de/erste-schritte/schnellstart) kommen die Punkte hinzu, die im Dauerbetrieb nötig werden: ein vollständiges Compose-Setup mit QGIS-Server und persistenten Volumes, die Absicherung über HTTPS, die Anbindung einer PostgreSQL/PostGIS-Datenbank sowie die Server-Einstellungen für Worker, Timeouts und Logging.

Die Beispiele sind so angelegt, dass Sie sie schrittweise übernehmen können; nicht jede gezeigte Option wird in jeder Installation gebraucht.

## docker-compose.yml

Ein ausführlicheres Beispiel als im [](/admin-de/erste-schritte/schnellstart). Nicht alle Optionen werden in jedem Fall benötigt; auskommentierte Zeilen sind optional.

```yaml title="docker-compose.yml"
services:
    gws:
        image: gbdconsult/gws-amd64:8.4
        container_name: gws
#       restart: unless-stopped
        ports:
            - "80:80"
#           - "443:443"
        volumes:
            - /var/gws/data:/data:ro
            - /var/gws-var:/gws-var
#           - /etc/letsencrypt/live/example.com:/data/ssl:ro
        tmpfs:
            - /tmp
        environment:
#           - GWS_CONFIG=/data/config.cx
#           - PGSERVICEFILE=/data/pg_service.conf
#           - GWS_LOG_LEVEL=INFO

    qgis:
        image: gbdconsult/gbd-qgis-server-amd64:3.44
        container_name: qgis
        volumes:
            - /var/gws/data:/data:ro
            - /var/gws-var:/gws-var
        tmpfs:
            - /tmp
```

## Images & Versionen

Die Images liegen unter https://hub.docker.com/u/gbdconsult, jeweils für `amd64` und `arm64`. Sie benötigen das WebSuite-Image (`gbdconsult/gws-<arch>`) und – für QGIS-Funktionen – das QGIS-Server-Image (`gbdconsult/gbd-qgis-server-<arch>`).

Ein Tag wie `8.4` zeigt stets auf den aktuellsten `8.4.x`-Stand. Releases sind nicht immer kompatibel; beim Wechsel auf eine neue Version (z. B. `8.4` → `8.5`) sind meist Anpassungen an der Konfiguration nötig.

## Volumes & Mounts

Im Beispiel werden Host-Verzeichnisse in die Container eingebunden. Bei Host-**Mounts** sind die Dateiberechtigungen relevant; ggf. müssen Sie `uid`/`gid` des Container-Users setzen (Default 1000/1000, siehe `GWS_UID`/`GWS_GID`).

- **`/data`** – [Konfiguration und Daten](/admin-de/erste-schritte/konfigurationsgrundlagen) (`.qgs`, GeoTIFF …). Weder WebSuite noch QGIS müssen hierhin schreiben; meist genügt `:ro`. Für große Rasterbestände empfiehlt sich ein eigenes Mount (z. B. `/data/raster`).
- **`/gws-var`** – Cache und Datenaustausch zwischen den Containern. Beide Container müssen hierhin schreiben können.
- **`tmpfs` `/tmp`** – temporäres Verzeichnis im Arbeitsspeicher.

## Ports

Die WebSuite antwortet auf `80/http` und `443/https`. Ist HTTPS über hinterlegte Zertifikate aktiv, wird ein permanenter Redirect von `http` auf `https` gesetzt.

%info
Manche Host-Firewalls (z. B. `ufw` unter Ubuntu) greifen bei von Docker weitergeleiteten Ports nicht – siehe [Docker-Dokumentation](https://docs.docker.com/network/packet-filtering-firewalls/).
%end

## Umgebungsvariablen

Umgebungsvariablen überschreiben stets den entsprechenden Eintrag in der Konfiguration.

| Variable | Config | Default | Beschreibung |
|---|---|---|---|
| `GWS_CONFIG` | – | `/data/config.cx` | Einstiegspunkt der Konfiguration |
| `GWS_LOG_LEVEL` | `server.log.level` | `INFO` | Log-Granularität (`ERROR`, `INFO`, `DEBUG`) |
| `PGSERVICEFILE` | – | – | Definitionsdatei benannter PostgreSQL-Verbindungen |
| `(HTTP\|HTTPS\|NO)_PROXY` | – | – | Proxy für ausgehende Anfragen |
| `GWS_UID` / `GWS_GID` | – | `1000` | UID/GID des Prozess-Users im Container |

Für den QGIS-Container gelten zusätzlich die [QGIS-Server-Variablen](https://docs.qgis.org/latest/en/docs/server_manual/config.html#environment-variables), u. a. `QGIS_WORKERS` (paralleles Rendern) und `PGSERVICEFILE`.

## Wichtige Konfigurationsthemen

Nicht alles ist über Umgebungsvariablen steuerbar. Die folgenden Einstellungen setzen Sie in der [Konfiguration](/admin-de/erste-schritte/konfigurationsgrundlagen):

- **Webserver & Rewrites** – mindestens eine `web.site` mit Rewrite-Regeln für lesbare URLs. Ein minimales Beispiel zeigt das [Einfache Projekt](/admin-de/erste-schritte/einfaches-projekt).
- **SSL** – Zertifikat und Schlüssel hinterlegen:

```js
web.ssl {
    crt "/data/ssl/example.com.crt"
    key "/data/ssl/example.com.key"
}
```

- **QGIS-Server** – Erreichbarkeit des `qgis`-Containers:

```js
server.qgis.host "qgis"
server.qgis.port 80
```

  Weitere Optionen: [gws.server.core.QgisConfig](/admin-de/reference/gws.server.core.QgisConfig).

- **PostgreSQL/PostGIS** – Datenbank-Provider in der Konfiguration hinterlegen:

```js
database.providers+ {
    type postgres
    uid "mydb"
    host "db.example.com"
    port 5432
    username "bob"
    password "*****"
}
```

  Der Einsatz von `pg_service.conf` wird empfohlen.

- **Locale** – für korrekte Beschriftungen, Zahlen- und Datumsformate:

```js
locales ["de_DE"]
```

- **Server-Einstellungen** – Worker/Threads, Timeouts, Logging und das automatische Neuladen der Konfiguration in [gws.server.core.Config](/admin-de/reference/gws.server.core.Config).

## Container steuern

Starten und Stoppen über `docker compose`:

    docker compose up -d      # starten
    docker ps                 # Status
    docker compose down       # stoppen

Befehle direkt an die Applikation senden Sie mit `docker exec`; eine Übersicht bietet die [Kommandozeilen-Referenz](/admin-de/konfiguration/cli):

    docker exec gws gws -h
