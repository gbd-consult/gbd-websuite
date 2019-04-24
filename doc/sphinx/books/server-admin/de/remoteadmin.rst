Remote-Verwaltung
=======================

Die GBD WebSuite kann von einem lokalen Computer aus verwaltet werden. Um die Remoteverwaltung nutzen zu können, benötigen Sie clientseitige Software wie unser QGIS-Plugin mit dem Sie einige Serverkonfigurationen vornehmen müssen.

Um die serverseitige Remoteverwaltung zu konfigurieren, müssen Sie eine Aktion (siehe: doc: `actions`) vom Typ ``remoteadmin`` unter ``api / actions`` aktivieren:

    "api": {
        "Aktionen":
            ...
            {
                "type": "remoteadmin"
            }


Die Remote-Verwaltung überspringt die normalen Autorisierungsmechanismen und verwendet das Kennwort, welches verschlüsselt in der Datei ``.remoteadmin`` im Server-Verzeichnis `` gws-var`` gespeichert wird (siehe: doc: `install`). Sie können den Standardpfad mit der Option `` passwordFile`` ändern. Verwenden Sie zum Erstellen der Kennwortdatei den Befehl `` auth passwd`` (siehe: doc: `cli`). ::

    gws auth passwd --path /gws-var/.remoteadmin
