Database
=========


Die GBD WebSuite kann Geodaten aus Datenbanken beziehen. Entweder indirekt über QGIS und WMS/WMTS-Anbieter oder durch die direkte Anbindung an eine Datenbank. Im letzteren Fall müssen Sie die Datenbankverbindungen in der Hauptkonfiguration einstellen.

Zur Zeit unterstützt die GBD WebSuite PostgreSQL/PostGIS Datenbanken. Die Unterstützung von SQLite/SpatiaLite, MySQL und MongoDB Datenbanken folgt in naher Zukunft.

Beispiel einer PostGIS-Provider-Konfiguration ::

    ## in der Hauptkonfiguration:

    "db": {
        "providers" [
            {
                "type": "postgis",
                "uid": "my_db_connection",
                "host": "DBHOST",
                "port": 5432,
                "database": "mydatabase",
                "user": "me",
                "password": "secret"
            }
        ]
    }


Wenn Sie mehrere Referenzen auf dem gleichen Server haben, müssen Sie diese als verschiedene Anbieter konfigurieren und jedem seine eigene ID zuweisen.

Wenn in der weiteren Konfiguration eine Datenbankverbindung erforderlich ist, muss lediglich die eindeutige ID des Anbieters angegeben werden. Beispiel für eine SQL-Suchkonfiguration::

        ## Suchkonfiguration

        {
            "type": "sql",
            "db": "my_db_connection",
            ...other options
        }


Da die GBD WebSuite in einem Container läuft, können Sie nicht ``localhost`` als Hostname verwenden, unabhängig davon ob Ihr DB-Server auf der selben physischen Maschine läuft. Stattdessen sollte die IP-Adresse des Docker-Hosts wie zum Beispiel ``172. 17. 0. 1`` verwendet werden (der genaue Wert hängt von den Einstellungen Ihres Docker-Netzwerks ab). Aus Gründen der Portabilität ist es empfehlenswert, es mit ``--add-host`` zu aliasieren.
