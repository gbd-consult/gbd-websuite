Installation
============

Vor der Installation
--------------------

Als Docker-Anwendung benötigt die GBD WebSuite keine Installation per se, aber Sie müssen ein paar Dinge einstellen, damit die GBD WebSuite problemfrei starten kann.

Verzeichnisse
~~~~~~~~~~~~~

Die GBD WebSuite benötigt einige Verzeichnisse, die von Ihrem Host-Rechner eingebunden werden müssen, um die Daten mit der Außenwelt auszutauschen.

- ein oder mehrere *Daten*-Verzeichnisse. Hier speichern Sie Ihre Konfiguration und Daten. Der Server schreibt nie in diese Verzeichnisse, daher ist es eine gute Idee, sie schreibgeschützt zu mounten. Diese Verzeichnisse können an beliebige Stellen im Container eingebunden werden (wir verwenden standardmäßig ``/data``).
- ein *var*-Verzeichnis, in dem der Server seine eigenen persistenten Daten wie Caches und Sitzungsdaten speichert. Es sollte an ``/gws-var`` im Container montiert werden.
- ein temporäres Verzeichnis *tmp*. Normalerweise würde man es als ``tmpfs`` bezeichnen.

Nutzer
~~~~~~

Die GBD WebSuite wird intern standardmäßig als Nutzer- und Gruppen-ID ``1000`` ausgeführt. Sorgen Sie bitte dafür, das dieser Nutzer in Ihrem Hostsystem

- über Leserechte für alle *Daten*-Verzeichnisse verfügt
- über Schreib- und Leserechte für *var* und *tmp* Verzeichnisse

Sie können die ``uid/gid``  Werte mit Umgebungsvariablen ``GWS_UID`` und ``GWS_GID`` ändern.

Port
~~~~

Die GBD WebSuite benutzt die Ports ``80`` und ``443``. Sie können sie auf alles abbilden, was Sie beim Testen wollen, und auf echte 80/443 in der Produktion.

Konfiguration
~~~~~~~~~~~~~

Die GBD WebSuite erwartet die Konfiguration in ``/data/config.json``. Wenn Sie einen anderen Speicherort und/oder ein anderes Format bevorzugen, setzen Sie die Umgebungsvariable ``GWS_CONFIG`` auf den Pfad Ihrer Konfigurationsdatei.

^SEE Die Konfigurationsformate sind unter ^config/intro beschrieben.

Externe Adressen
~~~~~~~~~~~~~~~~

Wenn Ihr GBD WebSuite Container externe Verbindungen benötigt (höchstwahrscheinlich zu Datenbankservern), benötigen Sie eine oder mehrere ``--add-host`` Optionen in Ihrem Docker-Startbefehl.

Einstiegspunkt
~~~~~~~~~~~~~~

Die GBD WebSuite hat einen einzigen Einstiegspunkt, ein Shell-Skript namens ``gws``. Um den Server zu starten oder zu stoppen, nutzen Sie die folgenden Befehle:

    gws server start
    gws server stop

Ausführen mit ``gws``-Shell-Skript
----------------------------

Folgende Optionen müseen Sie in Ihrem ``gws``-Shell-Skript anpassen:

- ein oder mehrere *Daten*-Verzeichnisse einbinden
- *var* und *tmp* Verzeichnisse einbinden
- Port-Mappings
- Konfigurationspfad
- externe Hosts

Hier ist eine Skript-Vorlage, die die GBD WebSuite mittels ``docker run`` startet. Sie können diese Vorlage nach Ihren Bedürfnissen anpassen. ::

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

Sobald Sie dieses Skript als z.B. ``gws`` in Ihren Pfad abspeichern, können Sie die WebSuite mit diesen Befehlen steuern: ::

    gws server start
    gws server stop
    gws server restart

Aktuellen Quellcode anbinden
----------------------------

Da die GBD WebSuite aktiv entwickelt wird, kann es vorkommen, dass eine in dem Docker-Image enthaltene Version von unserem Quellcode veraltet ist. Sie können aber das Image mit der aktuellen Version ausführen indem Sie das Quellcodeverzeichnis unter ``gws-app`` mounten.

Laden Sie zuerst unser Paket von Ihrer Version herunter (in diesem Fall, Version 6.1): ::

    curl -O http://gws-files.gbd-consult.de/gws-6.1.tar.gz

entpacken Sie das Paket: ::

    tar xvzf gws-6.1.tar.gz

und mounten Sie den ``gws-server/app`` Unterordner als ``gws-app``: ::

    --mount type=bind,src=<absoluter Pfad>/gws-server/app,dst=/gws-app,readonly

Host-Installation
-----------------

Wir haben auch ein Skript, das die WebSuite direkt auf Ihrem System installiert, ohne docker. Das Skript finden Sie in unserem Github unter https://github.com/gbd-consult/gbd-websuite/blob/master/install/install.sh

.. caution:: Diese Entwicklung ist experimentell, nicht auf Produktionsserver probieren!
