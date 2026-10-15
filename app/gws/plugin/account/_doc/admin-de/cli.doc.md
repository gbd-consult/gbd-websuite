# Kommandozeilen-Befehl "account" :/admin-de/konfiguration/cli/account

Die Befehlsgruppe `account` verwaltet Benutzerkonten von der Kommandozeile aus. `gws account reset --uid <ID>[,<ID>…]` setzt ein oder mehrere Konten zurück: Passwort und MFA-Geheimnis werden gelöscht, der Status wird auf *neu* gesetzt, und die Konten erhalten eine neue Freischaltungs-Mail, über die der Nutzer sein Passwort neu setzt (siehe [Benutzerkonten](/admin-de/themen/zugriff/konten)).

Den Befehl rufen Sie im Container auf, zum Beispiel `docker exec -it <container_name> gws account reset --uid 12`.
