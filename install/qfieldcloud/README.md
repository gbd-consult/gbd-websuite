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

Super user wird automatisch mit dem Benutzernamen:super_user und Email: super@user.com erstellt (beides kann in make.sh geändertwerden).

Nach dem Start ist die Web-Oberfläche unter http://private_IP_Adresse erreichbar.

Die MinIO-Konsole ist unter http://172.17.0.1:8009 verfügbar (Zugangsdaten sind einsehbar in der .env Datei).

Wie man sich mit der QfieldCloud verbindet findet man hier: https://docs.qfield.org/get-started/
