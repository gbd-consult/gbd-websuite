Installation
============

Als Docker-Anwendung benötigt die GBD WebSuite keine Installation per se, aber Sie müssen ein paar Dinge wissen, um es zum Laufen zu bringen.

Verzeichnisse
------------------

Die GBD WebSuite benötigt einige Verzeichnisse, die von Ihrem Host-Rechner eingebunden werden müssen, um die Daten mit der Außenwelt auszutauschen.

- ein oder mehrere "Daten"-Verzeichnisse. Hier speichern Sie Ihre Konfiguration und Daten. Der Server schreibt nie in diese Verzeichnisse, daher ist es eine gute Idee, sie schreibgeschützt zu mounten. Diese Verzeichnisse können an beliebige Stellen im Container eingebunden werden (wir verwenden standardmäßig ``/data``).
- das Verzeichnis "var", in dem der Server seine eigenen persistenten Daten wie Caches und Sitzungsdaten speichert. Es sollte an ``/gws-var`` im Container montiert werden.
- ein temporäres Verzeichnis. Normalerweise würde man es als ``tmpfs`` bezeichnen. 

Port
-----

Die GBD WebSuite zeigt die Ports "80" und "443" an. Sie können sie auf alles abbilden, was Sie beim Testen wollen, und auf echte ``80/443`` in der Produktion.

Hauptkonfiguration
-----------------------

Die GBD WebSuite erwartet seine Hauptkonfiguration in ``/data/config. json``. Wenn Sie einen anderen Speicherort und/oder ein anderes Format bevorzugen, setzen Sie die Umgebungsvariable ``GWS_CONFIG`` auf den Pfad Ihrer Hauptkonfiguration.

Externe Rechner
-------------------

Wenn Ihr GBD WebSuite Container externe Verbindungen benötigt (höchstwahrscheinlich zu Datenbankservern), benötigen Sie eine oder mehrere ``--add-host`` Optionen in Ihrem Docker-Startbefehl. 

Einstiegspunkt
--------------------------------

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

.. literalinclude:: /{SERVER_ROOT}/server-sample.sh
