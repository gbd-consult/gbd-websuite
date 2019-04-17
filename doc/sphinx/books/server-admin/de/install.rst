Installation
============

Als Docker-Anwendung benötigt die GBD WebSuite keine Installation per se, aber Sie müssen ein paar Dinge wissen, um sie zum Laufen zu bringen.

Verzeichnisse
-------------

Die GBD WebSuite benötigt einige Verzeichnisse, die von Ihrem Host-Rechner eingebunden werden müssen, um die Daten öffentlich zu stellen.

- Erstellen Sie ein oder mehrere "Daten"-Verzeichnisse. Hier speichern Sie Ihre Konfiguration und Daten. Der Server schreibt nie in diese Verzeichnisse, weshalb es sich anbietet, diese schreibgeschützt zu mounten. Die Verzeichnisse können an beliebige Stellen im Container eingebunden werden (wir verwenden standardmäßig ``/data``).
- Erstellen Sie das Verzeichnis "var", in dem der Server seine eigenen dauerhaften Daten wie Caches und Sitzungsdaten speichert. Es sollte an ``/gws-var`` im Container angehangen werden.
- Erstellen Sie ein temporäres Verzeichnis. Normalerweise würde man es als ``tmpfs`` bezeichnen aber das ist flexibel.

Port
-----

Die GBD WebSuite greift auf die Ports "80" und "443" zu. Sie können diese Ports für alles verwenden was Sie Testen wollen., und auf echte ``80/443`` in der Produktion.

Hauptkonfiguration
------------------

Die GBD WebSuite erwartet ihre Hauptkonfiguration in ``/data/config. json``. Wenn Sie einen anderen Speicherort und/oder ein anderes Format bevorzugen, setzen Sie die Umgebungsvariable ``GWS_CONFIG`` auf den Pfad Ihrer Hauptkonfiguration.

Externe Rechner
---------------

Wenn Ihr GBD WebSuite Container externe Verbindungen benötigt (höchstwahrscheinlich zu Datenbankservern), benötigen Sie eine oder mehrere ``--add-host`` Optionen in Ihrem Docker-Startbefehl.

Einstiegspunkt
--------------

Die GBD WebSuite hat einen einzigen Einstiegspunkt, ein Shell-Skript namens ``gws``. Um den Server zu starten und zu stoppen, möchten Sie einen von diesen ::

    gws server start
    gws server stop



Alles zusammensetzen
-----------------------

Also, hier sind Optionen, die Sie in Ihrem ``docker run`` Befehl anpassen müssen:

- eine oder mehrere "Daten"-Einbinden
- eine "var"-Einbinden
- eine "tmp"-Einbinden
- Port-Mappings
- Konfigurationspfad
- externe Hosts

Wir haben ein Beispielskript ``server-sample. sh``, das Sie an Ihre Bedürfnisse anpassen können:

.. literalinclude:: /{APP_DIR}/server-sample.sh
