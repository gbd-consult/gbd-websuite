# Kommandozeilen-Befehl "ows" :/admin-de/konfiguration/cli/ows

`gws ows caps` liest das Capabilities-Dokument eines externen OGC-Dienstes und gibt seinen Aufbau als JSON aus – nützlich, um die Ebenen und unterstützten Koordinatensysteme eines Dienstes vor dem Einbinden zu prüfen:

    gws ows caps --src https://example.com/wms --type WMS --out caps.json

`--src` ist die Dienst-Adresse oder eine XML-Datei, `--type` der Diensttyp (etwa `WMS`), `--out` die Ausgabedatei. Das Einbinden externer Dienste beschreibt das Thema [Externe Kartendienste](/admin-de/themen/karten/quellen).
