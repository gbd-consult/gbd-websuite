# Kommandozeilen-Befehl "qfieldcloud" :/admin-de/konfiguration/cli/qfieldcloud

Mit `gws qfieldcloud package` erzeugen Sie das Datenpaket eines QField-Projekts von Hand und legen es in einem Verzeichnis ab. Im laufenden Betrieb ist das nicht nötig – die WebSuite erstellt das Paket selbst, sobald QField es anfordert. Für Tests und zur Fehlersuche lässt sich der Vorgang so aber gezielt auslösen und das Ergebnis prüfen, ohne ein mobiles Gerät zu bemühen.

    gws qfieldcloud package --projectUid <Projekt> --qfcProjectUid <QField-Projekt> --dir <Zielverzeichnis>

`projectUid` benennt das Projekt der WebSuite, `qfcProjectUid` das darin konfigurierte QField-Projekt und `dir` das Verzeichnis, in das geschrieben wird. Sind mehrere `qfieldcloud`-Aktionen konfiguriert, wählen Sie die gewünschte mit `--actionName`; ohne Angabe wird `qfieldcloud` verwendet.

## Was im Zielverzeichnis landet

Das Paket enthält dieselben Bestandteile wie das, welches die App erhält: je editierbarem Layer ein GeoPackage, die kopierten Anhang- und Datenverzeichnisse sowie das umgeschriebene QGIS-Projekt unter `<QField-Projekt>.qgs`. Daneben legt die WebSuite zwei Dateien ab, die bei der Fehlersuche helfen:

- `<QField-Projekt>.qgs.source.qgs` – das unveränderte Ausgangsprojekt. Ein Vergleich mit der umgeschriebenen Fassung zeigt, welche Layer entfernt und welche Datenquellen ersetzt wurden.
- `path_map.json` – die Zuordnung der Dateinamen im Paket zu den tatsächlichen Pfaden auf dem Server.

Die gerenderte Hintergrundkarte liegt nicht im Zielverzeichnis, sondern im Kartencache des Projekts; `path_map.json` verweist darauf.

## Berechtigungen

Der Befehl läuft als Systembenutzer und nicht als eine der Feldkräfte. Er umgeht damit die Zugriffsregeln, die im Betrieb greifen: Ein Paket, das sich auf der Kommandozeile erzeugen lässt, muss für einen angemeldeten Nutzer noch lange nicht verfügbar sein. Prüfen Sie Berechtigungsfragen deshalb nicht mit diesem Befehl, sondern mit einer echten Anmeldung.

Die fachliche Einordnung gibt das [Fachthema QField / QFieldCloud](/admin-de/themen/fachmodule/qfieldcloud).
