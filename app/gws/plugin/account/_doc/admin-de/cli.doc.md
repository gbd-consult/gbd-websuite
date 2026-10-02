# Kommandozeilen-Befehl "account" :/admin-de/konfiguration/cli/account

Die Befehlsgruppe `account` verwaltet Benutzerkonten von der Kommandozeile aus. `gws account reset --uids <ID> [<ID> …]` setzt ein oder mehrere Konten zurück: Sie werden entsperrt und erhalten eine neue Freischaltungs-Mail, über die der Nutzer sein Passwort neu setzt (siehe [Benutzerkonten](/admin-de/themen/zugriff/konten)).

Den Befehl rufen Sie im Container auf, zum Beispiel `docker exec -it <container_name> gws account reset --uids 12`.
