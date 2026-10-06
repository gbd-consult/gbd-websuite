# Kommandozeilen-Befehl "exporter" :/admin-de/konfiguration/cli/exporter

Mit `gws exporter export` führen Sie einen Datenexport ohne Client aus – etwa für einen zeitgesteuerten Lauf. Der Befehl liest die Anfrage aus einer JSON-Datei und schreibt das Ergebnis an einen Pfad:

    gws exporter export --request request.json --output /pfad/export.zip

Die konzeptionelle Einordnung des Exports steht unter [Datenexport](/admin-de/themen/publishing/export).
