# Kommandozeilen-Befehl "gekos" :/admin-de/konfiguration/cli/gekos

Die Befehlsgruppe `gekos` verwaltet den GekoS-Index. Mit `gws gekos index` lesen Sie die Vorgänge aus Gek-Online neu ein und schreiben sie in die konfigurierte Zieltabelle. Mit `--projectUid` wählen Sie das Projekt, dessen `gekos`-Aktion verwendet wird.

Den Befehl rufen Sie im Container auf, zum Beispiel `docker exec -it <container_name> gws gekos index`.

%warn
Jeder Lauf legt die Zieltabelle neu an (siehe [Aktion "gekos"](/admin-de/konfiguration/action/gekos)). Ein bestehender Tabelleninhalt geht dabei verloren.
%end
