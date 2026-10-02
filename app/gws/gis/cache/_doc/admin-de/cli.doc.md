# Kommandozeilen-Befehl "cache" :/admin-de/konfiguration/cli/cache

Die Befehlsgruppe `cache` verwaltet den Kachel-Cache des Servers. Damit füllen Sie den Cache für bestimmte Layer vorab (`gws cache seed`), fragen mit `gws cache status` seinen Zustand ab und löschen mit `gws cache drop` oder `gws cache cleanup` aktive beziehungsweise veraltete Cache-Verzeichnisse. Die Befehle rufen Sie im Container auf, zum Beispiel `docker exec -it <container_name> gws cache status`.
