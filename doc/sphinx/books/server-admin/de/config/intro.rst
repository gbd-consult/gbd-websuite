Übersicht
=========

Die GBD WebSuite Konfiguration besteht aus Applikations-Konfigurationsdatein (^app) und optional mehrere Projekt-Konfigurationsdateien (^project). Die Struktur der Konfigurationsobjekte ist in ^../ref/config vollständig beschrieben.

Konfigurationsformate
---------------------

Die Konfirgurationen können in verschiedenen Sprachen geschrieben werden, nämlich JSON, YAML, SLON und Python. Sie können die Sprachen auch frei mischen, z.B. App-Konfig in Python und Projekt-Konfig in YAML.

JSON
~~~~

Bei JSON (https://www.json.org) handelt es sich um ein gängiges Konfigurations- und Datenaustauschformat. In dieser Dokumentation verwenden wir JSON für Code-Snippets und Beispiele. Dies ist auch unser Defaultformat: falls Sie keinen expliziten Konfigurationspfad mittels ``GWS_CONFIG`` bestimmen, wird eine JSON Datei ``config.json`` im "Data"-Verzeichnis geladen. JSON Konfigdateien müssen mit der Endung ``.json`` abgespeichert werden.

YAML
~~~~

YAML (https://yaml.org) ist eine Alternative zu JSON, die einfacher zu schreiben und lesen ist. Sie können Ihre Konfiguration in YAML schreiben, mit der gleichen Struktur wie JSON. YAML Konfigdateien müssen mit der Endung ``.yaml`` abgespeichert werden.

SLON
~~~~

SLON (https://github.com/gebrkn/slon) ist  eine Alternative zu JSON, die die Strukturen in einer weiter vereinfachten Form darstellt. Bei diesem Format können Sie auch alle Befehle der Templating-Sprache verwenden (wie z.B. ``@include`` oder ``@if``). Konfigdateien im SLON Format müssen eine Erweiterung ``.cx`` haben.

^SEE Mehr über Templating-Sprache lesen Sie unter ^template.

Python
~~~~~~

Komplexe, sich wiederholende oder hochdynamische Konfigurationen können auch direkt in def Programmiersprache Python geschrieben werden. Die Python-Konfigurationsdatei muss eine Funktion ``config()`` enthalten, die einen Python ``dict`` zurückgibt. Beachten Sie, dass Ihr Konfigurationsmodul innerhalb des Containers ausgeführt wird und daher mit Python 3.6 kompatibel sein muss.

Struktur der Konfiguration
--------------------------

Auf der obersten Ebene, ist die Konfiguration eine Schlüssel-Wert Struktur (*dict*), die Zeichenketten als Schlüssel und entweder die "primitiven" Werte (wie z.B. eine Zahl oder Zeichenkette) oder weitere Schlüssel-Wert Strukturen bzw. Listen (*arrays*) von Werten enthält.

Einige Schlüssel-Wert Strukturen haben eine grundlegende Eigenschaft Typ (``type``), der angibt, zu welchem Typ die gegebene Struktur gehört. Diese Eigenschaft ist stets anzugeben.

Eine weitere grundlegende Eigenschaft, def Identifikator (``uid``), ist dagegen Optional und ist nur dann anzugeben, wenn Sie auf die gegebene Struktur an weiteren Stellen der Konfiguration verweisen möchten. In anderen Fällen wird die ``uid`` aus dem Objekt-Titel bzw. Typ automatisch generiert.

Laden der Konfiguration
-----------------------

Die Konfiguration beginnt mit der App-Konfigurationsdatei (``GWS_CONFIG`` bzw ``data/config.json``). Sobald diese Datei erfolgreich gelesen ist, werden die Projekte geladen. Die Projekte werden mit drei Optionen in der App-Konfig bestimmt:

- ``projects`` (eine Liste von Projekten). Mit dieser Option werden Ihre Projekte direkt in der App-Konfig eingebunden.

- ``projectPaths`` (eine Liste von Dateinamen). Mit dieser Option werden Projekte aus angegebenen Dateien gelesen, wobei jede Datei ein Projekt oder eine Liste von Projekten enthält

- ``projectDirs`` (eine Liste von Ordnernamen). Mit dieser Option liest das System aus angegebenen Verzeichnissen alle Dateien die mit ``.config.json``, ``.config.yaml``, ``.config.cx`` oder ``.config.py`` enden und diese als Projekte konfiguriert.

Diese Optionen können miteinander auch frei kombiniert werden.

Monitoring
----------

GWS Server enthält ein *Monitor* Modul, der das Dateisystem überwacht, die Änderungen in Ihren Projekten und Konfigurationen überprüft und ggf. einen Hot-Reload des Servers durchführt. Sie können Intervalle für diese Prüfungen konfigurieren, es wird empfohlen, das Monitorintervall auf mindestens 30 Sekunden einzustellen, da Dateisystemprüfungen ressourcenintensiv sind.

^SEE Sie können Monitoring unter ^server konfigurieren.
