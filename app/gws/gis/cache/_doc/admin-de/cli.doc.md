# Kommandozeilen-Befehl "cache" :/admin-de/konfiguration/cli/cache

Die Befehlsgruppe `cache` verwaltet den Kachel-Cache des Servers. Damit füllen Sie den Cache für bestimmte Layer vorab (`gws cache seed`), fragen mit `gws cache status` seinen Zustand ab, löschen mit `gws cache drop` zwischengespeicherte Kacheln und entfernen mit `gws cache cleanup` verwaiste Cache-Verzeichnisse, die zu keinem konfigurierten Cache mehr gehören. Die Befehle rufen Sie im Container auf, zum Beispiel `docker exec -it <container_name> gws cache status`.
