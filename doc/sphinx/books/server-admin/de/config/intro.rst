Einführung in der Konfiguration
===============================

Die GWS Konfiguration besteht aus Hauptkonfigdatei und optional mehrere Projekt-Konfigurationsdateien.

Konfigurationsformate
---------------------

Diese Dateien können in verschiedenen Sprachen geschrieben werden, nämlich JSON, YAML, SLON und Python. Sie können die Sprachen auch frei mischen, z.B. Hauptkonfig in Python und Projekt-Konfig in YAML.

JSON
~~~~

JSON (https://www.json.org) ist ein gängiges Konfigurations- und Datenaustauschformat. In dieser Dokumentation verwenden wir JSON für Code-Snippets und Beispiele. Dies ist auch unser Defaultformat: falls Sie keinen expliziten Konfigurationspfad mittels ``GWS_CONFIG`` bestimmen, wird eine JSON Datei ``config.json`` im "Data"-Verzeichnis geladen. JSON Konfigdateien müssen mit dem Erweiterung ``.json`` abgespeichert werden.

YAML
~~~~

YAML (https://www.yaml.org) ist eine Alternative zu JSON, die einfacher zum Schreiben und Lesen ist. Sie können Ihre Konfiguration in YAML schreiben, mit der gleichen Struktur wie JSON. YAML Konfigdateien müssen mit dem Erweiterung ``.yaml`` abgespeichert werden.

SLON
~~~~

SLON (https://github.com/gebrkn/slon) ist  eine Alternative zu JSON, die die Strukturen in einer vereinfachten Form darstellt. Bei diesem Format können Sie auch alle Befehle der Templating-Sprache verwenden (wie z.B. ``@include`` oder ``@if``). Konfigdateien im SLON Format müssen eine Erweiterung ``.cx`` haben.

^SEE Mehr über Templating-Sprache lesen Sie unter ^format.

Python
~~~~~~

Komplexe, sich wiederholende oder hochdynamische Konfigurationen können auch direkt in Programmiersprache Python geschrieben werden. Die Python-Konfigurationsdatei muss eine Funktion ``config()`` enthalten, die einen Python ``dict`` zurückgibt, mit der  gleichen Struktur wie JSON. Beachten Sie, dass Ihr Konfigurationsmodul innerhalb des Containers ausgeführt wird und daher mit Python 3.6 kompatibel sein muss.

Laden der Konfiguration
-----------------------

Die Konfiguration beginnt mit der Hauptkonfigurationsdatei (``GWS_CONFIG`` bzw ``data/config.json``). Sobald diese Datei erfolgreich gelesen ist, werden die Projekte geladen. Die Projekte werden mit drei Optionen in der Hauptkonfig bestimmt:

- ``projects`` (eine Liste von Projekten). Mit dieser Option werden Ihre Projekte direkt in der Hauptkonfiguration eingebunden.

- ``projectPaths`` (eine Liste von Dateinamen). Mit dieser Option werden Projekte aus angegebenen Dateien gelesen, wobei jede Datei ein Projekt oder eine Liste von Projekten enthält

- ``projectDirs`` (eine Liste von Ordnernamen). Mit dieser Option liest das System aus angegebenen Verzeichnissen alle Dateien die mit ``.config.json``, ``.config.yaml``, ``.config.cx`` oder ``.config.py`` enden und diese als Projekte konfiguriert.

Diese Optionen können miteinander auch frei kombiniert werden.

Sobald alle Dateien gelesen und analysiert sind, werden alle konfigurierten Objekte zu einem großen "Baum" zusammengefasst, wobei das ``Application`` Objekt der Wurzelknoten ist. Hier ist ein Beispiel für einen solchen Baum::

   Application
    |
    |-- auth options
    |-- server options
    |-- web options
    |
    \-- projects
        |
        |-- First project
        |   |-- project options
        |   \-- Map
        |       |-- First layer
        |       \-- Second layer
        |
        \-- Second project
           |-- project options
           \-- Map
               \-- Layer group
                   \-- Sub-layer

Dieser Baum bleibt im Server-Speicher bis der GWS Server beendet oder neu konfiguriert wird.

Hauptkonfigurationsdatei
------------------------

^REF_DE gws.common.application.Config

Die wichtigsten Teile der Hauptkonfigurationsdatei sind die folgenden Abschnitte, die weiter im Detail beschrieben werden:

{TABLE}
    ``api`` |  Server-Aktionen (s. ^actions)
    ``auth`` |  Zugriffsrechte (s. ^auth)
    ``client`` |  Client Optionen (s. ^client)
    ``db`` |  Datenbankanbindung (s. ^db)
    ``locales`` |  Server Localen (s. ^intl)
    ``meta`` |  Globale Metadaten (s. ^meta)
    ``server`` |  Server Optionen (s. ^server)
    ``web`` |  Web-Server Konfigurationen (s. ^web)
{/TABLE}

Monitoring
----------

GWS Server beobachtet Änderungen in allen Konfigurationsdateien, Vorlagen und QGIS Projekten und wird neu konfiguriert sobald sich eine von diesen Dateien ändert. Das Neuladen des Servers erfolgt im Hintergrund.

^SEE Sie können Monitoring unter ^server konfigurieren.
