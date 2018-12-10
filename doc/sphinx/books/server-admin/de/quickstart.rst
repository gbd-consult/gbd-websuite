Schnellstart
============

Hiermit starten Sie den GBD WebSuite Server zum ersten Mal und richten Ihr erstes Projekt ein. 

**Schritt 1**. Stellen Sie sicher, dass `docker <https://www.docker.com>`_ installiert ist und funktioniert.

**Schritt 2**. Laden Sie das GBD WebSuite Server Image herunter und testen Sie es ::

    docker run -it -p 3333:80 --name my-gws-container gbdconsult/gws-server:latest

Dies führt den GBD WebSuite Server auf dem Port ``3333`` unter dem Namen ``my-gws-container`` aus (zögern Sie nicht, einen anderen Namen und/oder Port zu verwenden).
Bei einigen Setups benötigen Sie möglicherweise Root-Rechte (``sudo``), um dies ausführen zu können. Sie sollten das Serverprotokoll auf Ihrem Terminal sehen.

Wenn alles in Ordnung ist, sollten Sie das Server-Log auf Ihrem Terminal sehen. Gehen Sie in Ihrem Browser auf `<http://localhost:3333>`_. Es wird die Server-Startseite und unser Beispielprojekt gezeigt. 

Stoppen Sie nun den Server mit Control-C und entfernen Sie den Container ::

    docker rm my-gws-container

**Schritt 3**. Erstellen Sie ein Verzeichnis ``hello`` irgendwo auf Ihrer Festplatte (z. B. ``/var/work/hello``).
Erstellen Sie in diesem Verzeichnis eine Datei namens ``config. json`` mit folgendem Inhalt ::


    {
        "title": "Hello",
        "map": {
            "view": {
                "extent": [554000, 6461000, 954000, 6861000],
                "scales": [1e3, 5e3, 1e4]
            },
            "layers": [
                {
                    "title": "OpenStreetMap",
                    "type": "client",
                    "kind": "osm"
                }
            ]
        }
    }



**Schritt 4**. Führen Sie den Container erneut aus und geben Sie ihm einen Pfad zu Ihrer neu erstellten Konfiguration ::

    docker run -it -p 3333:80 --name my-gws-container --mount type=bind,src=/var/work/hello,dst=/data/projects/hello gbdconsult/gws-server:latest

Navigieren Sie zu ``http://localhost:3333/hello <http://localhost:3333/hello>``. Sie sollten die OpenStreetMap von Düsseldorf sehen, der Geburtsort der GBD WebSuite. 


