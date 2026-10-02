# Datenbank-Provider "postgres" :/admin-de/konfiguration/databaseProvider/postgres

Dieser Provider stellt eine Verbindung zu einer PostgreSQL-Datenbank mit PostGIS-Erweiterung her. Er wird von Layern, Modellen und Suchprovidern genutzt, die auf Tabellen dieser Datenbank zugreifen.

Sie können die Verbindung entweder direkt über `host`, `database`, `username` und `password` angeben oder – empfohlen – über `serviceName` auf einen in der `pg_service.conf` definierten Dienst verweisen. So halten Sie die Zugangsdaten aus der Konfiguration heraus. Über `options` lassen sich zusätzliche libpq-Verbindungsparameter setzen.

## Beispiel-Konfiguration über einen Dienst (empfohlen) ::

```javascript
database.providers+ {
    uid "main_db"
    type "postgres"
    serviceName "LOCAL"
}
```

Der Provider wird über `database.providers+` aktiviert. `serviceName` verweist auf einen in der `pg_service.conf` definierten Dienst, der Host, Datenbank und Zugangsdaten enthält. So bleiben die Zugangsdaten aus der Konfiguration heraus. Die `uid` dient anderen Objekten (etwa über `dbUid`) als Verweis auf diesen Provider.

## Beispiel-Konfiguration mit direkten Zugangsdaten ::

```javascript
database.providers+ {
    uid "main_db"
    type "postgres"
    host "localhost"
    port 5432
    database "gws"
    username "gws"
    password "secret"
    schemaCacheLifeTime "3600"
}
```

Alternativ geben Sie die Verbindung direkt über `host`, `port`, `database`, `username` und `password` an. `schemaCacheLifeTime` legt fest, wie lange die eingelesenen Tabellenschemata zwischengespeichert werden.

%ref "gws.plugin.postgres.provider.Config"
