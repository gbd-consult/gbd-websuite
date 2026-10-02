# Kommandozeilen-Befehl "auth" :/admin-de/konfiguration/cli/auth

Die Befehlsgruppe `auth` verwaltet die Authentifizierungs-Sitzungen des Servers. Mit `gws auth sessions` zeigen Sie die aktiven Sitzungen an, mit `gws auth sessrem` entfernen Sie Sitzungen, etwa alle abgelaufenen oder gezielt einzelne. Die Befehle rufen Sie im Container auf, zum Beispiel `docker exec -it <container_name> gws auth sessions`.

Mit `gws auth password` erzeugen Sie zu einem eingegebenen Passwort den verschlüsselten Wert, den Sie in die Benutzerdatei des dateibasierten Authentifizierungs-Providers eintragen.
