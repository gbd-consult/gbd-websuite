# Kommandozeilen-Befehl "server" :/admin-de/konfiguration/cli/server

Die Befehlsgruppe `server` steuert den GBD-WebSuite-Server. Damit starten Sie den Server (`gws server start`), starten ihn neu (`gws server reload`) oder konfigurieren ihn neu (`gws server reconfigure`); mit `gws server configtest` prüfen Sie die Konfiguration, ohne den Server anzufassen. Die Befehle rufen Sie im Container auf, zum Beispiel `docker exec -it <container_name> gws server configtest`.
