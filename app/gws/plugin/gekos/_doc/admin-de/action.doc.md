# Aktion "gekos" :/admin-de/konfiguration/action/gekos

Die Aktion `gekos` stellt die Schnittstelle zur Fachanwendung GekoS bereit. Sie liefert zu Flurstücks- und Adress-Codes die zugehörigen Koordinaten und Objektdaten und verbindet GekoS so mit der Flurstückssuche. Mit `index` konfigurieren Sie den GekoS-Index, mit `templates` die Vorlagen für die Objektdarstellung.

## GIS-Schnittstelle in GekoS einrichten

Damit beide Systeme aufeinander verweisen, tragen Sie die Adressen der WebSuite im GekoS-Programm unter *Verfahrensadministration → GIS-Schnittstelle* ein. Ersetzen Sie dabei `PROJECT_ID` durch die uid Ihres WebSuite-Projekts; die spitzen Klammern sind Platzhalter, die GekoS selbst füllt.

| Einstellung | Wert |
|---|---|
| `GIS-URL-Base` | `http://mein-server` |
| `GIS-URL-ShowXY` | `/project/PROJECT_ID/?x=<x>&y=<y>&z=SCALE_VALUE` |
| `GIS-URL-ShowFs` | `/project/PROJECT_ID/?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>` |
| `GIS-URL-GetXYFromMap` | `/project/PROJECT_ID/?x=<x>&y=<y>&gekosUrl=<returl>` |
| `GIS-URL-GetXYFromFs` | `/_/gekosGetXY/projectUid/PROJECT_ID/fs/<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>` |
| `GIS-URL-GetXYFromGrd` | `/_/gekosGetXY/projectUid/PROJECT_ID/ad/<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>` |

Die Adressen zerfallen in zwei Gruppen. Die ersten drei öffnen die Karte im Browser: `ShowXY` setzt eine Markierung auf die übergebene Koordinate, `ShowFs` stellt ein Flurstück dar, und `GetXYFromMap` lässt den Nutzer einen Punkt wählen und schickt ihn an die in `gekosUrl` genannte Adresse zurück. Sie werden vom Client verarbeitet und setzen voraus, dass das Werkzeug `Toolbar.Gekos` in der [Client-Konfiguration](/admin-de/konfiguration/client) eingetragen ist.

Die beiden `GetXY`-Adressen sind dagegen Rückrufe ohne Oberfläche: Sie sprechen unmittelbar die Aktion an und liefern als Antwort reinen Text im Format `x;y` mit drei Nachkommastellen, im Fehlerfall `error:`. Damit sie funktionieren, muss im Projekt auch die Aktion `alkis` verfügbar sein – die Auflösung der Codes übernimmt das ALKIS-Modul.

%warn
Die Reihenfolge der Platzhalter in den zusammengesetzten Codes ist verbindlich und muss den Feldern des ALKIS-Moduls entsprechen. Eine abweichende Reihenfolge führt nicht zu einer Fehlermeldung, sondern zu einem leeren Suchergebnis.
%end

## Import aus Gek-Online :index

Über das `index`-Objekt übernimmt die WebSuite Vorgänge aus dem Modul Gek-Online in eine räumliche Datenbanktabelle. Unter `sources` geben Sie je Quelle die Adresse (`url`), die Anfrageparameter (`params`) und eine Kennung (`instance`) an; `dbUid`, `tableName` und `crs` bestimmen Ziel und Koordinatensystem. Ausgeführt wird der Abgleich über die Kommandozeile, siehe [Kommandozeilen-Befehl "gekos"](/admin-de/konfiguration/cli/gekos).

Über `position` verschieben Sie die eingelesenen Punkte (`offsetX`, `offsetY`); liegen mehrere Vorgänge auf derselben Koordinate, ordnet die WebSuite sie über `distance` und `angle` kreisförmig an, damit sie im Client unterscheidbar bleiben.

%warn
Der Import legt die Zieltabelle bei jedem Lauf **neu an** – die vorhandene Tabelle wird gelöscht und mit einem festen Spaltenschema neu erstellt. Verwenden Sie eine eigene Tabelle, die ausschließlich diesem Zweck dient. Datensätze der Gek-Online-Quelle ohne `X`/`Y`/`ObjectID` werden übersprungen. Die Abfrage der Quelle prüft das TLS-Zertifikat nicht.
%end

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "gekos"
    index {
        dbUid "DB_GEKOS"
        tableName "gekos.vorgaenge"
        crs 25832
        sources+ {
            url "https://gekos-online.example.de/gekos"
            instance "gekos1"
            params {
                Version "5"
            }
        }
        position {
            offsetX 0
            offsetY 0
            distance 2
            angle 90
        }
    }
}
```

`index` konfiguriert den Import aus Gek-Online. Unter `sources` geben Sie je Quelle die Adresse (`url`), die Anfrageparameter (`params`) und eine Kennung (`instance`) an; `dbUid`, `tableName` und `crs` bestimmen Ziel und Koordinatensystem. Über `position` verschieben Sie die eingelesenen Punkte (`offsetX`, `offsetY`) und ordnen Vorgänge auf gleicher Koordinate über `distance` und `angle` kreisförmig an. Der Abgleich selbst läuft über die Kommandozeile.

%ref "gws.plugin.gekos.action.Config"
