# Kommandozeilen-Befehl "printer" :/admin-de/konfiguration/cli/printer

Mit `gws printer print` erzeugen Sie ein Druckdokument ohne Client. Der Befehl liest den Druckauftrag aus einer JSON-Datei und schreibt das Ergebnis an einen Pfad:

    gws printer print --request request.json --output /pfad/druck.pdf

Das Projekt nennt der Auftrag selbst in `projectUid`; die Option `--project` wird nicht ausgewertet.

Wie das Drucken abläuft, beschreibt das Thema [Drucken](/admin-de/themen/darstellung/drucken).
