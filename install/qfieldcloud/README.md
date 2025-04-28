## QFieldCloud lokale Installation

Dieses Verzeichnis enthält Skripte zur einfachen lokalen Installation und Ausführung von QFieldCloud.

### Voraussetzungen

- Docker und Docker Compose
- Git
- Bash-Shell

### Verwendung

1. QFieldCloud Repository klonen:
   ```
   ./make.sh clone
   ```

2. Docker-Images bauen:
   ```
   ./make.sh build
   ```

3. QFieldCloud starten:
   ```
   ./make.sh up
   ```

4. QFieldCloud stoppen:
   ```
   ./make.sh down
   ```

5. Shell im Web-Container öffnen:
   ```
   ./make.sh bash
   ```

6. Aufräumen:
   ```
   ./make.sh clean
   ```

Nach dem Start ist die Web-Oberfläche unter http://localhost:8000 erreichbar.

Die MinIO-Konsole ist unter http://localhost:9001 verfügbar (Zugangsdaten: minioadmin/minioadmin).
