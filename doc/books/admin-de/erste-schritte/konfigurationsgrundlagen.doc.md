# Konfigurationsgrundlagen :/admin-de/erste-schritte/konfigurationsgrundlagen

Die Konfiguration bestimmt das gesamte Verhalten der WebSuite. Dieses Kapitel erklärt Syntax und Aufbau; das [Einfache Projekt](/admin-de/erste-schritte/einfaches-projekt) wendet beides praktisch an.

Im Kern ist die Konfiguration ein einziges JSON-Objekt. In der Praxis schreiben Sie sie jedoch selten in reinem JSON, sondern nutzen die beiden mitgelieferten Präprozessoren: *slon* lässt überflüssige Satzzeichen weg, *jump* ergänzt Templating und das Aufteilen auf mehrere Dateien. Beide werden vor dem Einlesen wieder zu JSON aufgelöst; die Struktur bleibt also dieselbe.

Dieses Kapitel beschreibt zunächst beide Schreibweisen und anschließend, wo die WebSuite ihre Konfiguration sucht und wie sich die Dateien sinnvoll auf ein Verzeichnis verteilen lassen. Welche Objekte und Eigenschaften es gibt, ist nicht Gegenstand dieses Kapitels, sondern der [](/admin-de/konfiguration).

## Syntax

Im Kern ist die Konfiguration ein [JSON](https://www.json.org/json-de.html)-Objekt. Sie können die gesamte Konfiguration in reinem JSON schreiben (Einstiegspunkt z. B. `config.json`). Bequemer sind die beiden mitgelieferten Präprozessoren **slon** und **jump** – die Dokumentation nutzt sie durchgängig. Beide werden vor dem Einlesen wieder zu JSON aufgelöst.

### slon

*slon* („simple JSON") lässt überflüssige Syntax weg:

- Kommata zwischen Listen- oder `key value`-Einträgen entfallen, wenn Leerzeichen oder Zeilenumbruch trennen.
- Anführungszeichen um Schlüssel und um einwortige Zeichenketten entfallen, ebenso der Doppelpunkt.

```javascript
{
    access "allow all"
    actions [
        { type web }
        { type map }
    ]
}
```

Zusätzlich:

- `objekt.schlüssel wert` setzt bzw. ergänzt Werte in (auch bestehenden) Objekten, z. B. `provider.maxRequests 4`.
- `liste+ eintrag` hängt einen Eintrag an eine Liste an:

```javascript
{
    access "allow all"
    actions+ { type web }
    actions+ { type map }
    projects+ {
        uid myproject
        title "Mein Testprojekt"
        map.layers+ {
            type tile
            title OSM
            provider.url "https://osmtiles.gbd-consult.de/ows/{{z}}/{{x}}/{{y}}.png"
        }
    }
}
```

### jump

*jump* ergänzt Templating und das Aufteilen der Konfiguration auf mehrere Dateien.

`@include` fügt den Inhalt einer Datei an dieser Stelle ein:

```javascript title="config.cx"
{
    access "allow all"
    actions+ { type web }
    actions+ { type map }
    projects [
        @include /data/config/projects/myproject.cx
    ]
}
```

Geschweifte Klammern: Folgt auf `{` ein Leerzeichen oder Zeilenumbruch, gilt sie wie gewohnt. Folgt kein Leerzeichen, beginnt ein Ausdruck (Templating). Wird eine `{` ohne folgendes Leerzeichen als Literal gebraucht, muss sie verdoppelt werden – siehe die `{{z}}/{{x}}/{{y}}`-Platzhalter oben.

## Einstiegspunkt & Struktur

Die WebSuite liest zuerst die über `GWS_CONFIG` definierte Datei (Default `/data/config.cx`); alle weiteren Dateien werden von dort eingebunden. Wie Sie die Dateien aufteilen, bleibt Ihnen überlassen. Empfohlen und in der Dokumentation verwendet wird folgende Struktur:

```
data
├── assets                  Eigene Vorlagen, falls die mitgelieferten nicht genügen
├── config
│   ├── projects            Projektspezifische Konfigurationen
│   │   └── myproject.cx
│   ├── web.cx
│   └── client.cx
├── config.cx               Einstiegspunkt
├── qgis                    QGIS-Projektdateien
│   └── myqgisproject.qgs
├── users.json              Dateibasierte Benutzer und Rollen
└── web                     Statische Assets (Logo, CSS …)
```
