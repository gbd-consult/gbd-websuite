# Host-Konfiguration :/admin-de/konfiguration/host

Dieser Abschnitt beschreibt die Konfiguration auf dem Host – also alles, was außerhalb der WebSuite-Konfiguration liegt: die Container, ihre Verzeichnisse und Netzwerkanbindung sowie die Datenbankverbindung.

Die [konzeptionelle Einordnung](/admin-de/themen/betrieb/host) und der [schrittweise Aufbau einer Installation](/admin-de/erste-schritte/installation) stehen in eigenen Kapiteln.

## docker-compose.yml

Eine vollständige Installation besteht aus dem WebSuite-Container und – sobald QGIS-Projekte im Spiel sind – einem QGIS-Server-Container. Beide teilen sich dieselben Verzeichnisse.

```yaml title="docker-compose.yml"
services:
    gws:
        image: gbdconsult/gws-amd64:8.4
        container_name: gws
        restart: unless-stopped
        volumes:
            - ./data:/data:ro
            - ./gws-var:/gws-var
        ports:
            - "80:80"
            - "443:443"
        tmpfs:
            - /tmp
        environment:
            - GWS_CONFIG=/data/config.cx
            - PGSERVICEFILE=/data/pg_service.conf
            - GWS_LOG_LEVEL=INFO

    qgis:
        image: gbdconsult/gbd-qgis-server-amd64:3.44
        container_name: qgis
        restart: unless-stopped
        volumes:
            - ./data:/data:ro
            - ./gws-var:/gws-var
        tmpfs:
            - /tmp
        environment:
            - PGSERVICEFILE=/data/pg_service.conf
```

Beachten Sie dabei:

- **`restart: unless-stopped` in beiden Containern.** Docker startet den Container damit nach
  einem Absturz und nach einem Neustart des Hosts von selbst wieder — nicht aber, wenn Sie ihn
  ausdrücklich mit `docker compose stop` angehalten haben. Für Wartungsarbeiten bleibt er also
  aus, bis Sie ihn selbst wieder starten. Setzen Sie die Angabe bei **beiden** Diensten: Fehlt
  sie beim QGIS-Server, läuft die WebSuite nach einem Neustart des Hosts zwar, aber alle Layer
  aus QGIS-Projekten bleiben leer.
- **`/data` schreibgeschützt einbinden.** Die WebSuite schreibt nie in dieses Verzeichnis; `:ro` verhindert versehentliche Änderungen zur Laufzeit.
- **`/gws-var` muss beschreibbar und dauerhaft sein.** Dort liegen Caches, Sitzungen und weitere Laufzeitdaten. Ein Container-Neustart darf dieses Verzeichnis nicht leeren.
- **`tmpfs` für `/tmp`.** Temporäre Dateien landen so im Arbeitsspeicher des Hosts und nicht auf der Festplatte.
- **`PGSERVICEFILE` in beiden Containern setzen.** Sonst findet der QGIS-Server die Datenbankverbindungen nicht, obwohl die WebSuite sie kennt.

## Umgebungsvariablen

Sie setzen die Variablen im `environment`-Block der Container. Wo eine Konfigurationsoption dasselbe steuert, hat die Umgebungsvariable Vorrang.

| Variable | Bedeutung | Voreinstellung |
|---|---|---|
| `GWS_CONFIG` | Pfad zur Einstiegsdatei der Konfiguration | `/data/config.cx` |
| `GWS_GID` | Gruppen-ID des Server-Nutzers | `1000` |
| `GWS_LOG_LEVEL` | Ausführlichkeit des Logs (`ERROR`, `INFO`, `DEBUG`) | `INFO` |
| `GWS_MANIFEST` | Pfad zur Manifest-Datei für Plugins | – |
| `GWS_SPOOL_WORKERS` | Anzahl der Spool-Prozesse | – |
| `GWS_TMP_DIR` | Temporäres Verzeichnis | `/tmp/gws` |
| `GWS_UID` | Benutzer-ID des Server-Nutzers | `1000` |
| `GWS_VAR_DIR` | Verzeichnis für veränderliche Laufzeitdaten | `/gws-var` |
| `GWS_WEB_WORKERS` | Anzahl der Web-Prozesse | – |
| `PGSERVICEFILE` | Pfad zur `pg_service.conf` | – |

## Logging

Wie ausführlich die WebSuite protokolliert, steuert `GWS_LOG_LEVEL`. Wohin die Ausgaben gehen, entscheidet dagegen Docker, nicht die WebSuite.

Ohne weitere Angabe verwenden die Container den Standard-Treiber; die Ausgaben lesen Sie dann mit `docker logs gws` bzw. `docker logs qgis`. Mit einem `logging`-Block leiten Sie sie stattdessen an das Syslog des Hosts weiter und versehen sie mit einem Tag, nach dem sich filtern lässt:

```yaml title="docker-compose.yml"
services:
    gws:
        logging:
            driver: syslog
            options:
                tag: GWS_APP
    qgis:
        logging:
            driver: syslog
            options:
                tag: GWS_QGIS
```

Beide Container schreiben dann nach `/var/log/syslog`; `tail -f /var/log/syslog | grep GWS_` zeigt die Ausgaben beider Container gemeinsam.

## Datenbankverbindungen: pg_service.conf

Ein Datenbank-Provider kann Zugangsdaten unmittelbar enthalten. Empfohlen ist stattdessen eine `pg_service.conf`: Die Konfiguration nennt dann nur noch den Dienstnamen, die Zugangsdaten bleiben außerhalb – und QGIS und die WebSuite nutzen dieselbe Datei.

```ini title="pg_service.conf"
[mydb]
host=db.example.com
port=5432
dbname=meine_daten
user=bob
password=geheim

[alkis]
host=db.example.com
port=5432
dbname=alkis
user=alkis_ro
password=geheim
```

In der Konfiguration genügt dann der Dienstname:

```javascript
database.providers+ {
    type postgres
    uid "mydb"
    serviceName "mydb"
}
```

%warn
Die Datei muss UNIX-Zeilenenden verwenden. Eine unter Windows erstellte `pg_service.conf` mit CRLF wird nicht korrekt gelesen, und die Verbindung schlägt ohne aussagekräftige Meldung fehl.
%end

## Datenbank auf dem Host erreichen

Läuft PostgreSQL nicht als Container, sondern unmittelbar auf dem Host, ist `localhost` **nicht** die richtige Adresse: Innerhalb des Containers verweist `localhost` auf den Container selbst.

Tragen Sie stattdessen einen Namen für den Host ein und lassen Sie Docker ihn auflösen:

```yaml title="docker-compose.yml"
services:
    gws:
        extra_hosts:
            - "host.docker.internal:host-gateway"
```

In der `pg_service.conf` verwenden Sie dann `host=host.docker.internal`. Alternativ geben Sie die IP-Adresse der Docker-Bridge an, üblicherweise `172.17.0.1`.

Denken Sie daran, in `pg_hba.conf` und `postgresql.conf` Verbindungen aus dem Docker-Netz zuzulassen.
