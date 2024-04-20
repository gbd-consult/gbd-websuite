# PostgreSQL :/admin-de/themen/postgresql

Der Zugriff auf PostgreSQL findet durch unterschiedliche Applikationen statt:

1. Durch die GBD WebSuite direkt
2. Durch den QGIS Server
3. Durch QGIS Desktop
4. Durch ein Datenbankmanagement Tool

Alle diese Applikationen benötigen einige Informationen um sich mit der 
PostgreSQL Datenbank verbinden zu können:

- **host**: Entweder eine IP Adresse oder einen Hostnamen (z.B. eine Domain)
  der zu der IP Adresse des Hosts aufgelöst werden kann.
- **port**: Standard ist 5432, ein numerischer Wert der bei der Einrichtung 
  des Datenbanksystems festgelegt werden kann.
- **database/dbname**: Auf einem Host läuft ein Datenbankcluster, welches mehrere 
  Datenbanken beinhalten kann. Der Name der Datenbank muss angegeben werden.
- **username**: Der Name des Benutzers mit dem wir uns auf die Datenbank 
  verbinden dürfen.
- **password** Das Passwort des Benutzers.
- _Alternativ_: **servicename**, siehe [pg_service.conf](/admin-de/themen/postgresql/pg_service.conf)

### Datenbankprovider :/admin-de/themen/postgresql/provider

Muss die GBD WebSuite sich mit der Datenbank direkt verbinden, z.B. um eine 
[postgres-Layer](TODO LINK) anzuzeigen, in der Datenbank zu [Suchen](TODO LINK),
um [Modelle](TODO LINK) zu [editieren](TODO LINK) oder gegen die Datenbank zu 
[authentifizieren](TODO LINK), so muss ein Datenbankprovider in der Konfiguration 
der GBD WebSuite hinterlegt werden, der eine Verbindungsmöglichkeit zu der 
Datenbank beschreibt.

#### config

```javascript
{
    database.providers+ {
        type postgres
        uid "example_db"
        host "db.example.com"
        port 5432
        database "my_gis_db"
        username "exampleuser"
        password "examplepassword123"
    }
}
```

Die **uid** ist ein von der GBD WebSuite intern verwendeter Bezeichner für
eine spezifische Datenbankverbindung. Jeder Provider muss eine einzigartige `uid`
erhalten. Existieren mehrere Provider muss bei der Konfiguration eines 
[Modells](TODO LINK) die hier vergebene `uid` angegeben werden um den zu 
verwendenden Provider eindeutig zu indentifizieren.

#### options

Es ist möglich ein options Dictionary zu hinterlegen um die LibPQ eigenen 
Parameter direkt zu setzen:
[https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS)

Beispiel:
```
{
    database.providers+ {
        type postgres
        ...
        options {
            "connect_timeout": 10
            "passfile": "/data/.pgpass"
        }
    }
}
```

#### schemaCacheLifeTime

Die GBD WebSuite merkt sich die Struktur der Datenbank um Abfragen effizienter 
konstruieren zu können. Standardmäßig überprüft die GBD WebSuite alle 3600 Sekunden
ob sich die Datenbankstruktur geändert hat.


### pg_service.conf :/admin-de/themen/postgresql/pg_service.conf

Greift nicht nur die GBD WebSuite, sondern auch der QGIS Server, oder Anwender 
mit QGIS Desktop auf die Datenbank zu reicht in einem Unternehmen mit mehreren 
mitarbeitern schnell der gemeinsame Zugriff über einen einzelnen 
Datenbankbenutzer nicht mehr aus.

Im folgenden wird ein Szenario beschrieben welches durch die Verwendung der
pg_service.conf zur Ablage von PostgreSQL Verbindungen stark vereinfacht werden 
kann:

- Es gibt eine PostgreSQL Datenbank mit mehreren Benutzerrollen, die mit 
  teils unterschiedlichen Berechtigungen auf eine oder mehrere gemeinsame 
  Datenbanken zugreifen müssen.
- Die Zugansinformationen zur Datenbank können somit nicht direkt in QGIS 
  Projekten (.qgs) abgelegt werden, da durch Weitergabe der QGIS Projekte 
  zwischen Anwendern diese Ihre Zugangsinformationen zur Datenbank mit weitergeben
  würden.
- Die QGIS Projekte werden in einer Datenbank abgelegt. Einerseits können die 
  Anwender so auf die gleichen Projekte zugreifen ohne eine gemeinsame Dateiablage
  zu verwenden, andererseits liest die GBD WebSuite diese QGIS Projekte aus der 
  Datenbank um diese direkt als Karten im Browser oder als Dienst zur Verfügung 
  zu stellen.

Dieses Szenario kann wie folgt gelöst werden:

im Home-Verzeichnis jedes Anwenders, sowie im /data/ Verzeichnis der GBD WebSuite
wird eine Textdatei (`pg_service.conf`) hinterlegt, welche mindestens einen Dienst 
definiert:


{file /data/pg_service.conf}
```
[mein_dienst]
host=db.example.com
port=5432
dbname=example_db
user=exampleuser
password=******

[mein_anderer_dienst]
host=db.example.com
port=5432
dbname=noch_eine_datenbank
user=readonly_user
password=*******
```

%info
Die `pg_service.conf` muss **immer** (auch auf Windows) UNIX Zeilenumbrüche 
verwenden.
%end

In der `docker-compose.yml` muss die Umgebungsvariable `PGSERVICECONF` für 
sowohl den `gws` als auch den `qgis` Container gesetzt werden.

Werden QGIS Projekte aus der Datenbank geladen, muss für den `qgis` Container 
ebenfalls die Variable `QGIS_SERVER_PROJECT_CACHE_STRATEGY` auf `periodic` 
gesetzt werden, damit Änderungen an den QGIS Projekten vom QGIS Server erkannt 
werden.

{file docker-compose.yml}
```yaml
services:
  gws:
    ...
    environment:
      - PGSERVICEFILE=/data/pg_service.conf
  qgis:
    ...
    environment:
      - PGSERVICEFILE=/data/pg_service.conf
      - QGIS_SERVER_PROJECT_CACHE_STRATEGY=periodic
```

Die Konfiguration der `database.providers` für die GBD WebSuite kann ebenfalls 
die Zugangsdaten aus der `/data/pg_service.conf` lesen:

{file /data/config.cx}
```javascript
{
  database.providers+ {
    type postgres
    uid mein_dienst
    service mein_dienst
  }
  database.providers+ {
    type postgres
    uid db2
    service mein_anderer_dienst
  }
}
```

Für die Verwendung mit Desktop QGIS kann die `pg_service.conf` an einem 
beliebigen Ort im Dateisystem des Desktop Rechners hinterlegt werden. 
Hier kann für jeden Anwender ein eigener User zum Verbindungsaufbau mit der 
Datenbank hinterlegt werden. Wird die `pg_service.conf` durch zentralisierte 
Administration automatisch für den Anwender hinterlegt sind hier sogar 
regelmäßige Passwort rotationen für Postgres-Anwenderaccounts möglich ohne dass 
der Anwender davon etwas mitbekommt.
Für QGIS Desktop muss die `PGSERVICEFILE` Umgebungsvariable ebenfalls gesetzt werden. 

Bei der Anlage der Datenbankverbindung in QGIS muss dann lediglich der Name des 
Dienstes angegeben werden, in diesem Beispiel `mein_dienst` oder `mein_anderer_dienst`.
Dies ist die einzige Information die im QGIS Projekt (`.qgs`) für eine Datenquelle 
hinterlegt wird.

### QGIS Projekte in Datenbank

QGIS kann Projekte direkt in der Datenbank speichern, statt in einer `.qgs`-Datei.

Die GBD WebSuite kann für die Darstellung von [QGIS-Layern](TODO LINK) diese 
QGIS Projekte direkt aus der Datenbank lesen.

### host.docker.internal

Ist PostgreSQL auf dem Host System direkt installiert, so kann in der 
[Providerkonfiguration](/admin-de/themen/postgresql/config) oder der 
[pg_service.conf] nicht der Hostname `localhost` verwendet werden.

Wird in der `docker-compose.yml` der folgende Eintrag hinterlegt:

{file docker-compose.yml}
```yaml
services:
  gws:
    ...
    extra_hosts:
      - "host.docker.internal:host-gateway"
  qgis:
    ...
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

ist es möglich unter `host.docker.internal` auf den Hostrechner des Containers 
zuzugreifen.

Dieser Name muss dann auch in der `pg_service.conf` oder der Providerkonfiguration 
als Hostname hinterlegt werden.

`host.docker.internal` ist der konventionell für den Zugriff des Containers auf 
den Hostrechner verwendete Name. Wenn gewünscht kann dieser an allen Stellen durch 
einen selbst gewählten Namen ersetzt werden.

