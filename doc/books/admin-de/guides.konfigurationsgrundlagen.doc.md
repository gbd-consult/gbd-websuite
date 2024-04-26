# Konfigurationsgrundlagen :/admin-de/konfigurationsgrundlagen

<!--
    Schreibstil:
        Dokumentation, Fakten, Gründe
        Leser muss verstehen
        Links zu detaillierterer 
        Keine Direkte Ansprache des Lesers
    
    Inhalte:
        - syntax: json, slon, jump
        - einstiegspunkt
        - verzeichnisstruktur beispiel
        - datentypen, listen, dicts
        - konfigurationsstruktur
        - @include
        - list+
-->

## Syntax (json + templating)

Die Konfiguration der Applikation ist ein 
[JSON](https://www.json.org/json-de.html) Objekt (`{}`). Es ist möglich den 
Einstigspunkt auf z.B. `config.json` zu setzen und die gesamte Konfiguration 
ausschließlich in JSON zu schreiben.

Um das Schreiben der Konfiguration weniger umständlich zu gestalten ermöglicht 
die GBD WebSuite die Verwendung von zwei Präprozessoren: [slon]() und [jump]().
Die Verwendung dieser wird empfohlen, und alle Beispiele in der Dokumentation
verwenden Features dieser wenn angebracht.

Die slon und jump Beispiele werden stets zu dem folgenden JSON Objekt umgewandelt 
welches daraufhin zur Konfiguration der GBD WebSuite verwendet wird:

{/gws-var/config/data_config_cx.parsed.json}
```json
{
    "access": "allow all",
    "actions": [
        { "type": "web" },
        { "type": "map" }
    ],
    "projects": [
        {
            "uid": "myproject",
            "title": "Mein Testprojekt",
            "map": {
                "layers": [
                    {
                        "type": "tile",
                        "title": "OSM",
                        "provider": { "url": "https://osmtiles.gbd-consult.de/ows/{z}/{x}/{y}.png" }
                    }
                ]
            }
        }
    ]
}
```

### slon (simple json)

[slon]() vereinfacht zunächst das schreiben von JSON, indem unnötige Syntaxelemente
weggelassen werden können.
- Kommata zwischen Elementen einer Liste fallen weg solange mindestens ein 
  Leerzeichen oder ein Zeilenumbruch diese trennt.
- Kommata zwischen `"key": "value"`-Paaren fallen weg solange mindesten in 
  Leerzeichen oder ein Zeilenumbruch als Trennung dient.
- Anführungszeichen um den Key von `key: "value"`-Paaren fallen weg.
- Doppelpunkte zwischen `key "value"`-Paaren fallen weg.
- Anführungszeichen um `strings` fallen weg, solange diese aus nur einem Wort bestehen.

```javascript
{
    access "allow all"
    actions [
        { type web }
        { type map }
    ]
    projects [
        {
            uid myproject
            title "Mein Testprojekt"
            map {
                layers [
                    {
                        type tile
                        title OSM
                        provider { url "https://osmtiles.gbd-consult.de/ows/{z}/{x}/{y}.png" }
                    }
                ]
            }
        }
    ]
}
```

- Weiterhin ermöglicht slon das Setzen von Werten in einem Objekt wie folgt: `object.key value`.
  Dies kann auch Eigenschaften in bereits existierenden Objekten ergänzen:

```javascript title="config.slon"
...
layers [
    {
        type tile
        title OSM
        provider.url "https://osmtiles.gbd-consult.de/ows/{z}/{x}/{y}.png"
        provider.maxRequests 4
    }
]
...
```

- Schließlich kann mit `liste+ eintrag` ein Eintrag in eine Liste eingefügt werden.

```javascript title="config.slon"
{
    access "allow all"
    actions+ { type web }
    projects+ {
        uid myproject
        title "Mein Testprojekt"
        map.layers+ {
            type tile
            title OSM
            provider.url "https://osmtiles.gbd-consult.de/ows/{z}/{x}/{y}.png"
        }
    }
    actions+ { type map }
}
```

### jump (templating)

Als nächster Schritt ermöglicht [jump]() die Verwendung von Templates und 
Aufteilen der Config auf mehrere Dateien.

Zunächst sind die wichtigsten Veränderungen die jump zur Konfiguration bringt:
#### @include

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

Dies ersetzt die Zeile `@include /pfad/zu/datei.cx` durch den Inhalt 
dieser Datei.

#### Verändertes Verhalten von `{}` geschweiften Klammern:

- Eine geschweifte Klammer auf die ein Leerzeichen oder Zeilenumbruch folgt
  wird wie vorher auch interpretiert.
- Eine geschweifte Klammer auf die _kein_ Leerzeichen oder Zeilenumbruch folgt
  beginnt einen [Ausdruck](TODO LINK ZU ERWEITERTE KONFIGURATION).
- Wird eine geschweifte Klammer ohne folgendes Leerzeichen/Zeilenumbruch 
  benötigt, muss diese wie im folgenden Beispiel gedoppelt werden

```javascript title="/data/config/projects/myproject.cx"
{
    uid myproject
    title "Mein Testprojekt"
    map.layers+ {
        type tile
        title OSM
        provider.url "https://osmtiles.gbd-consult.de/ows/{{z}}/{{x}}/{{y}}.png"
    }
}

```

Einen tieferen Einblick in die Möglichkeiten mit jump (z.B. Templates) gibt es 
im Guide [Fortgeschrittene Konfiguration](/admin-de/fortgeschrittene-konfiguration).


## Struktur

Dem System-/GIS-Administrator der GBD WebSuite sind sie größtmöglichen Freiheiten 
gelassen wie er die Konfiguration strukturiert.

Die GBD WebSuite wird stets die durch die Umgebungsvariable `GWS_CONFIG` definierte
Datei lesen (Default: `/data/config.cx`). Alle weiteren Dateien werden durch diese 
Datei eingebunden.

Es wird die folgende Struktur empfohlen, und in der Dokumentation verwendet:

```
.
├── data                            
│   ├── assets                      Dynamische Assets
│   │   ├── index.cx.html           Startseite mit Login & Projektliste
│   │   └── project.cx.html         Projektseite mit Kartenapplikation
│   ├── config
│   │   ├── actions.cx              
│   │   ├── auth.cx                 
│   │   ├── projects                Verzeichnis für projektspezifische Konfigs
│   │   │   └── myproject.cx        Konfig für das Projekt 'myproject'
│   │   └── web.cx                  
│   ├── config.cx                   Einstiegspunkt der Konfiguration
│   ├── MANIFEST.json               
│   ├── plugins                     Verzeichnis für Plugins
│   ├── qgis                        Verzeichnis für QGIS Projektdateien
│   │   └── myqgisproject.qgs       QGIS Projektdatei
│   ├── templates                   Verz. für dynamische Vorlagen
│   ├── users.json                  Dateibasierte Benutzerkonten und Rollen
│   └── web                         Statische Assets
│       ├── logo.png                
│       └── styles.css
└── docker-compose.yml              Containerkonfiguration
```

Der inhaltliche Aufbau der Konfigurationsdateien wird im Guide [Einfaches Projekt](/admin-de/einfaches-projekt) beschrieben.

