# postgres :/admin-de/config/layer/type/

^REF gws.ext.layer.postgres.Config

Vektor-Layer, übernimmt Geometrien aus einer Postgres-Tabelle. Sie müssen die zu verwendende Tabelle angeben

        {
            "title": "Houses",
            "type": "postgres",
            "table": {
                "name": "myschema.mytable"
            },
            "loadingStrategy": "bbox",
            "style": {
                "type": "css",
                "text": "stroke:rgb(0,255,0); stroke-width:1px; fill:rgba(0,100,0,0.2)"
            }
        }

Mehr über Tabellen-Konfigurationen siehe in ^db.


# postgres :/admin-de/config/db/type/

^REF gws.ext.db.provider.postgres.Config

Wir unterstützen PostgreSQL ab Version 10 mit der obligatorisch PostGIS (ab 2.4) Extension.

Beispiel einer Postgres-Provider-Konfiguration

    "db": {
        "providers" [
            {
                "type": "postgres",
                "uid": "my_db",
                "host": "172.17.0.1",
                "port": 5432,
                "database": "mydatabase",
                "user": "me",
                "password": "secret"
            }
        ]
    }

Wenn Sie mehrere Server bzw. mehrere Zugangsdaten auf dem gleichen Server haben, müssen Sie diese als verschiedene Anbieter konfigurieren.

^NOTE Auch wenn Ihr DB-Server sich auf demselben physischen Host befindet, können Sie nicht `localhost` als Hostname verwenden, weil die GBD WebSuite in einem Docker-Container läuft. Stattdessen sollte die IP-Adresse des Docker-Hosts wie `172.17.0.1` verwendet werden (der genaue Wert hängt von den Einstellungen Ihres Docker-Netzwerks ab). Aus Gründen der Portabilität ist es empfehlenswert, es mit `--add-host` zu aliasieren.
