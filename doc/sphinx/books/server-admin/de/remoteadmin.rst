Remote-Verwaltung
=================

Die GBD WebSuite kann per remote von einem lokalen Computer aus verwaltet werden. Um die Remoteverwaltung nutzen zu können, benötigen Sie clientseitige Software wie unser QGIS-Plugin und einige Serverkonfigurationen.

Um die serverseitige Remoteverwaltung zu konfigurieren, müssen Sie folgende Aktion vom Typ ``remoteadmin`` unter ``api/actions``aktivieren (siehe :doc:`actions`). ::

    "api": {
        "actions":
            ...
            {
                "type": "remoteadmin"
            }


Die Remoteverwaltung überspringt normale Autorisierungsmechanismen und verwendet das Kennwort, welches verschlüsselt in der Datei ``.remoteadmin`` auf dem Server ``gws-var`` gespeichert ist (siehe :doc:`install`). Sie können den Standardpfad in den `` passwordFile``Optionen ändern. Verwenden Sie den Befehl  ``auth passwd`` um die Kennwortdatei zu erstellen (siehe :doc:`cli`) ::

    gws auth passwd --path /gws-var/.remoteadmin
