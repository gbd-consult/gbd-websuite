# Kommandozeilen-Befehl "auth" :/admin-de/konfiguration/cli/auth

Die Befehlsgruppe `auth` verwaltet die Authentifizierungs-Sitzungen des Servers. Mit `gws auth sessions` zeigen Sie die aktiven Sitzungen an, mit `gws auth sessrem` entfernen Sie Sitzungen: alle (`--all`), die seit mehr als der angegebenen Zahl von Sekunden unbenutzten (`--older`) oder gezielt einzelne (`--uid`). Die Befehle rufen Sie im Container auf, zum Beispiel `docker exec -it <container_name> gws auth sessions`.

Mit `gws auth password` erzeugen Sie zu einem eingegebenen Passwort den Passwort-Hash, den Sie in die Benutzerdatei des dateibasierten Authentifizierungs-Providers eintragen.
