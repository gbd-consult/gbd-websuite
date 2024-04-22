# Installation :/admin-de/installation

Dieser Guide behandelt die Installation der GBD WebSuite, nicht des unterliegenden Betriebssystems oder Docker.

## Container Config

### docker-compose.yml

Dies ist ein möglichst umfangreiches Beispiel für eine `docker-compose.yml`. Es werden in den meisten Fällen keineswegs alle hier demonstrierten Einstellungsmöglichkeiten benötigt. Die einzelnen Optionen werden im Verlauf dieses Abschnitts näher erläutert. Auskommtierte Zeilen sind initiell optional.

{file docker-compose.yml}
```yaml
version: '3'

services:
    gws:
        image: gbdconsult/gws-amd64:8.0
        container_name: gws
#       restart: unless-stopped
#       logging:
#           driver: syslog
#           options:
#               tag: GWS_APP
        ports:
            - "80:80"
#           - "443:443"
        volumes:
            - /var/gws/data:/data:ro
            - /var/gws-var:/gws-var
#           - /var/raster:/data/raster:ro
#           - /etc/letsencrypt/live/webgis.example.com:/data/ssl:ro
        tmpfs:
            - /tmp
        environment:
#           - MANIFEST=/data/MANIFEST.json
#           - GWS_CONFIG=/data/config.cx
#           - PGSERVICEFILE=/data/pg_service.conf
#           - GWS_LOG_LEVEL=INFO
#           - HTTP_PROXY=http://proxy.example.com:8080
#           - HTTPS_PROXY=http://proxy.example.com:8080
#           - NO_PROXY=localhost
#       extra_hosts:
#           - "host.docker.internal:host-gateway"
#       depends_on:
#           - qgis

    qgis:
        image: gbdconsult/qgis-amd64:8.0
        container_name: qgis
#       restart: unless-stopped
#       logging:
#           driver: syslog
#           options:
#               tag: GWS_QGIS
        volumes:
            - /var/gws/data:/data:ro
            - /var/gws-var:/gws-var
#           - /var/raster:/data/raster:ro
        tmpfs:
            - /tmp
        environment:
#           - PGSERVICEFILE=/data/pg_service.conf
#           - QGIS_SERVER_PROJECT_CACHE_STRATEGY=periodic
#           - QGIS_DEBUG=0
#           - QGIS_WORKERS=4
#           - SVG_PATHS=/data/web/svg
#           - HTTP_PROXY=http://proxy.example.com:8080
#           - HTTPS_PROXY=http://proxy.example.com:8080
#           - NO_PROXY=localhost
#       extra_hosts:
#           - "host.docker.internal:host-gateway"
```

### Images & Versions

%include snippets/release-info.md

Die Images für die GBD WebSuite sind unter https://hub.docker.com/u/gbdconsult zu finden.
Die relevanten Images sind: `gws-amd64`, `qgis-amd64`, `gws-arm64` und `qgis-arm64`.

Es stehen Images für `amd64` sowie `arm` zur Verfügung.

Version `8.0` zeigt stets auf die aktuellste `8.0.x`.
Sobald der nächste Release (z.B. `8.1`) stabil ist, wird dies auf https://gbd-websuite.de/help.html angekündigt.
Releases sind nicht immer Kompatibel, es ist also zu erwarten, dass Sie ein paar Änderungen an der Konfiguration vornehmen müssen wenn Sie von z.B. `8.0` auf `8.1` upgraden.

Es gibt sowohl ein Image für die GBD WebSuite `gbdconsult/gws-<arch>:<version>`, sowie für den unterstüzenden QGIS-Server (`gbdconsult/qgis-<arch>:<version>`). Für die Verfügbarkeit einiger Features müssen beide Images laufen.


### Volumes & Mounts

Im `docker-compose.yml` Beispiel werden Verzeichnisse vom Host-Dateisystem in den Container eingebunden. 

Sollen **Volumes** verwendet werden, muss für den Anwender der die Konfiguration in `/data` schreibt eine Zugriffsmöglichkeit durch einen weiteren Container geschaffen werden welcher scp/sftp/smb zur Verfügung stellt.

Für die im Beispiel verwendete Möglichkeit mit Dateisystem **Mounts**, ist zu beachten dass die Berechtigungen im Host Dateisystem relevant sind. Dafür kann es nötig sein die `uid`/`gid` (Default: 1000/1000) des Users im Container explizit zu setzen. Siehe [Umgebungsvariablen](/admin-de/guides/installation/environment).

#### /data

Das `/data` Verzeichnis in beiden Containern enthält die Konfiguration sowie Daten (.qgs, .geotiff, ...) für die GBD WebSuite. Der Inhalt dieses Verzeichnisses wird näher unter [Konfigurationsgrundlagen](/admin-de/guides/konfigurationsgrundlagen) erläutert.

Weder die GBD WebSuite, noch der QGIS Server müssen in diesem Verzeichnis schreiben. In den meisten fällen kann es in der `docker-compose.yml` auf readonly `:ro` gesetzt werden.

So lange sich in diesem Verzeichnis keine großen Mengen an z.B. Rasterdaten befinden empfiehlt es sich den Inhalt des Verzeichnisses mit z.B. [`git`](https://git-scm.com/docs/git) zu Versionieren.

Für größere Mengen an Rasterdaten kann ein weiteres Verzeichnis gemounted werden, z.B. nach `/data/raster`. Damit kann dieses Verzeichnis leicht auf ein einges Volume ausgelagert werden.

#### /gws-var

Dieses Verzeichnis wird zum Cachen unterschiedlichster Daten so wie zum Austausch mancher Daten zwischen beiden Containern genutzt.

Beide Container müssen auf dieses Verzeichnis schreiben können.

#### tmpfs

Stellt dem Container ein Verzeichnis `/tmp` zur Verfügung, welches lediglich im Arbeitsspeicher des Hosts existiert. Mehr dazu hier: https://docs.docker.com/storage/tmpfs/

#### /gws-app

Enhält den Sourcecode der Applikation. Der Sourcecode wird im jeweiligen Container mitgeliefert und muss (außer von Entwicklern) nicht gemounted werden.


### Ports

Die GBD WebSuite antwortet auf HTTP und HTTPS Anfragen auf den Standardports `80/http` und `443/https`. 
Haben Sie durch Hinterlegen von Zertifikaten in der Konfiguration HTTPS aktiviert, so wird ein _permanenter_ Redirect auf `https` hinterlegt.

%info
[Manche Firewalls auf Host Rechnern (z.B. ufw auf Ubuntu) werden unter Umständen von Docker für auf den Host weitergeleitete Ports ignoriert.](https://docs.docker.com/network/packet-filtering-firewalls/)
%end


### Logging

Ohne den auskommentierten Logging Block im Beispiel wird das Logging von Docker verwendet, und die Logs sind über `docker logs gws` bzw. `docker logs qgis` abrufbar.

Wird der Logging Block einkommentiert, werden alle Ausgaben in die Datei `/var/log/syslog` weitergeleitet und mit einem entsprechenden Tag versehen nach dem gefiltert werden kann. z.B. `sudo tail -f /var/log/syslog | grep "GWS_"`.

Dies sind Features von Docker, nicht von der GBD WebSuite. Weitere Informationen dazu sind in der Dokumentation von Docker zu finden.


### Umgebungsvariablen

#### gws

Einige der Umgebungsvariablen für den `gws`-Container sind auch stattdessen direkt in der Konfiguration setzbar.
Umgebungsvariablen überschreiben stets den Eintrag in der Konfiguration.

<table>
<tr>
    <th>Variable</th>
    <th>Config</th>
    <th>Default</th>
    <th>Beschreibung</th>
</tr>
<tr>
    <td>MANIFEST</td>
    <td>-</td>
    <td>/data/MANIFEST.json</td>
    <td>TODO LINK [MANIFEST.json]()</td>
</tr>
<tr>
    <td>GWS_CONFIG</td>
    <td>-</td>
    <td>/data/config.cx</td>
    <td>Einstiegspunkt der Konfiguration</td>
</tr>
<tr>
    <td>PGSERVICEFILE</td>
    <td>-</td>
    <td></td>
    <td>Definition benannter PostgreSQL Connections (Dienste)</td>
</tr>
<tr>
    <td>GWS_LOG_LEVEL</td>
    <td>server.log.level</td>
    <td>'ERROR'</td>
    <td>Steuert granularität der Logs ('ERROR', 'INFO', 'DEBUG')</td>
</tr>
<tr>
    <td>(HTTP|HTTPS|NO)_PROXY</td>
    <td></td>
    <td></td>
    <td>Proxy der für ausgehende Anfragen verwendet wird.</td>
</tr>
<tr>
    <td>GWS_UID</td>
    <td>-</td>
    <td>1000</td>
    <td>Definiere die UID des Users welcher im Container den GWS Prozess ausführt.</td>
</tr>
<tr>
    <td>GWS_GID</td>
    <td>-</td>
    <td>1000</td>
    <td>Definiere die GID des Users welcher im Container den GWS Prozess ausführt.</td>
</tr>
</table>

#### qgis

Eine Vollständige Liste von Umgebungsvariablen die zur Konfiguration des QGIS Servers verwendbar sind finden Sie hier:
https://docs.qgis.org/latest/en/docs/server_manual/config.html#environment-variables

Zusätzlich dazu kann die Variable `PGSERVICEFILE` ebenfalls direkt an den Container übergeben werden.

Erwähnenswerte Variablen sind:

<table>
<tr>
    <td>PGSERVICEFILE</td>
    <td>Definitionsdatei benannter PostgreSQL Connections (Dienste)</td>
</tr>
<tr>
    <td>QGIS_WORKERS</td>
    <td>paralleles Rendern mehrerer Anfragen (limitiert durch CPU Kerne)</td>
</tr>
<tr>
    <td>QGIS_SERVER_PROJECT_CACHE_STRATEGY</td>
    <td>automatische Aktualisieren von in PostGIS abgelegten QGIS Projekten</td>
</tr>
<tr>
    <td>SVG_PATHS</td>
    <td>Pfade in denen nach Symbologie gesucht wird</td>
</tr>
</table>


## GWS Config

Nicht alle Einstellungsmöglichkeiten sind über Umgebungsvariables kontrollierbar. Es wird empfohlen einen Blick in [Konfigurationsgrundlagen](/admin-de/guides/konfigurationsgrundlagen) zu werfen um zu Verstehen wie die folgenden Optionen gesetzt werden können.

Im Folgenden eine Liste an Themen die Sie als Systemadministrator entweder für Ihren
GIS Administrator/Fachanwender konfigurieren müssen oder bei denen Sie mit diesem 
zusammenarbeiten müssen:


### WebServer, URLS, Rewrites

Der interne WebServer (nginx) des `gws`-Containers benötigt mindestens eine 
Konfigurierte `site` mit entsprechenden Rewrites um die einzelnen Seiten und 
Karten der Applikation unter leserlichen URLs anzeigen zu können.

Eine minimale Beispielhafte Konfiguration finden Sie im TODO LINK starterguide

Eine Auführliche Dokumentation finden Sie im Thema TODO LINK (WEBSERVER SITES URLS REWRITES)


#### SSL

Aktivieren von SSL (und damit automatischer _permanenter_ Redirect von `http` 
auf `https`) geschieht durch Hinterlegen eines Zertifikat+Schlüssel Paares:

```js
web.ssl {
    crt "/data/ssl/example.com.crt"
    key "/data/ssl/example.com.key"
}
```


### QGIS Server 

In dem `docker-compose.yml` Beispiel ist der `qgis`-Container für den `gws`-Container
wie folgt zu erreichen. Falls Sie die Konfiguration modifizieren müssen sind 
dies die relevanten Einträge in der `/data/config.cx`:

```js
server.qgis.host "qgis"
server.qgis.port 80
```

Diverse Umgebungsvariablen für den `qgis`-Container sind auch hier setzbar, siehe
[gws.server.core.QgisConfig](/admin-de/reference/gws.server.core.QgisConfig)


### PostgreSQL/PostGIS Datenbank

Damit die GBD WebSuite auf die Datenbank zugreifen kann muss ein Datenbank 
Provider in der `/data/config.cx` hinterlegt werden. Hostname muss dem `gws`-Container
bekannt sein, evtl `extra_hosts` Eintrag in der `docker-compose.yml` ergänzen (für beide Container).

```js
database.providers+ {
    type postgres
    uid "mydb" // beliebig setzbar, kann an anderen orten in der config dazu verwendet werden einen db provider eindeutig zu identifizieren
    host "db.example.com"
    port 5432
    username "bob"
    password "*****"
}
```

Es wird sehr empfohlen die Verwendung von `pg_service.conf` in Betracht zu ziehen, mehr dazu hier: #TODO link zu thema pg_service, qgis-projekte in postgis


### Locale

Damit in der Benutzeroberfläche richtige Beschriftungen angezeigt werden, sowie 
Korrekte Zahlen- und Datumsformate ist ein locale zu setzen:

```js
locales ["de_DE"]
```


### Server Settings

Es wird empfohlen einmal durch [gws.server.core.Config](/admin-de/reference/gws.server.core.Config)
zu schauen um einen Überblick darüber zu erhalten welche Einstellungen noch 
interessant sein können.

z.B. Anzahl Worker/Threads, Timeouts, Timezones, maxRequestLengths, logging, 
FS-Monitoring & automatic Config reloading, ...


## Applikationsteuerung

Starten und Stoppen des Containers geschieht mittels `docker compose`:

Starten:
```
docker compose -f docker-compose.yml up -d
```

Status anzeigen:
```
docker ps
CONTAINER ID   IMAGE                       ...  STATUS        PORTS                                      NAMES
c1d432be6d3b   gbdconsult/qgis-amd64:8.0   ...  Up 2 months                                              qgis
35b93c8834a1   gbdconsult/gws-amd64:8.0    ...  Up 2 months   0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp   gws
```

Stoppen:
```
docker compose -f docker-compose.yml down
```

Um Befehle direkt an die Applikation innerhalb des Containers zu senden: 
[Kommandozeilenreferenz](/admin-de/cli)

```
$ docker exec gws gws -h

GWS version 8.0.27
~~~~~~~~~~~~~~~~~~

    gws action invoke      - Invoke a web action.
    gws action profile     - Profile a web action.
    gws alkis index        - Create the ALKIS index.
    gws auth password      - Encode a password for the authorization file
    gws auth sessions      - Print currently active sessions
    gws cache cleanup      - Remove stale cache dirs.
    gws cache drop         - Remove active cache dirs.
    gws cache status       - Display the cache status.
    gws gekos index        - Create the GEKOS index.
    gws ows caps           - Print the capabilities of a service in JSON format
    gws printer print      - Print using the specified params
    gws qgis caps          - Print the capabilities of a document in JSON format
    gws qgis dbread        - Copy a project from the db to a local file.
    gws qgis dbwrite       - Copy a project from a local file to the db.
    gws server configtest  - Test the configuration.
    gws server configure   - Configure the server, but do not restart.
    gws server reconfigure - Reconfigure and restart the server.
    gws server reload      - Restart the server.
    gws server start       - Configure and start the server.

Try "gws <command> -h" for more info.
```
