Datenbanken
===========

Die GBD WebSuite kann Geodaten aus Datenbanken beziehen, entweder indirekt über QGIS und WMS/WMTS-Anbieter oder durch direkte Anbindung an eine Datenbank. Im letzteren Fall müssen Sie die Datenbankverbindungen in der Hauptanwendung config konfigurieren.

Zur Zeit unterstützen wir nur PostgreSQL/PostGIS Datenbanken. Wir planen, Sqlite/SpatiaLite, MySQL und MongoDB später hinzuzufügen.

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

Wenn Sie mehrere Credentials auf dem gleichen Server haben, müssen Sie diese als verschiedene Anbieter konfigurieren.

An anderer Stelle, wenn für Ihre Konfiguration eine Datenbankverbindung erforderlich ist, geben Sie einfach die eindeutige ID des Anbieters an. Beispiel für eine SQL-Suchkonfiguration::

        ## Suchkonfiguration

        {
            "type": "sql",
            "db": "my_db_connection",
            ...other options
        }

Da GWS in einem Container läuft, können Sie nicht ``localhost`` als Hostname verwenden, auch wenn Ihr DB-Server auf derselben physischen Maschine läuft. Stattdessen sollte die IP-Adresse des Docker-Hosts wie ``172. 17. 0. 1`` verwendet werden (der genaue Wert hängt von den Einstellungen Ihres Docker-Netzwerks ab). Aus Gründen der Portabilität ist es empfehlenswert, es mit ``--add-host`` zu aliasieren.
