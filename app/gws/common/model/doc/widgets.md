## Widgets


Mit der Option `widget` kann man Formularelemente für Felder konfigurieren. Die Anzeige und Verhalten von Widgets sind von Zugriffsrechten abhängig:

- wird kein `widget` definiert, wird auch kein Formularelement angezeigt

- wenn Nutzer über kein Leserecht verfügt, wird kein Formularelement angezeigt

- wenn Nutzer über kein Schreibrecht verfügt, wird das Formularelement als `readonly` angezeigt

- ansonsten wird das Formularelement normal angezeigt


Beachten Sie: wenn Sie das Widget weglassen, wird damit das direkte Aktualisieren des Feldes durch API nicht verhindert. Sollte das Feld nicht schreibbar sein, müssen dafür Zugriffsrechte definiert werden!

Es steht eine Reihe von Widgets zur Verfügung. Für machen Widgets können zusätzliche Optionen konfiguriert werden.

### `staticText`

Statischer, nicht editierbarer Text, für Felder die nur zur Information vorhanden sind.

### `input`

Eine allgemeine Input-Box.

```
widget { type "input" }
```

### `textArea`

Ein allgemeines `textarea` Element. Mit der Option `height` kann die logische Höhe (Anzahl von Zeilen) angegeben werden.

```
widget { type "textArea" height 10 }
```

### `integerInput`

Eine Input-Box zur Eingabe einer Ganzzahl. Sind `min`, `max` und `step` konfiguriert, werden auch erhöhen/vermindern Buttons angezeigt.

```
widget { type "integer" min 0 max 100 step 2 }
```

### `floatInput`

Eine Input-Box zur Eingabe einer reellen Zahl. Die Option `precision` gibt an, wie viele Nachkommastellen angezeigt werden.

```
widget { type "float" precision 3 min 0 max 1 step 0.05 }
```

### `date`, `time`, `dateTime`

Widgets zur Eingabe von Datum, Uhrzeit und Datum-Uhrzeit Werten.

```
widget { type "date" }
```

### `file`

Widget für `file` und `blob` Felder, mit Bildvorschau.

```
widget { type "file" }
```

### `select`

Eine Liste der Optionen mit Werten.

```
widget {
    type "select"
    items [
        [1, "Frau"],
        [2, "Herr"],
        [3, "Sonstiges"]
    ]
}
```

### `comboBox`

Eine Liste der string Optionen mit der Möglichkeit einer freien Eingabe

```
widget {
    type "comboBox"
    items [
        "Mit freundlichen Grüßen",
        "Viele Grüße",
        "Liebe Grüße"
    ]
}
```

### `checkBox`, `radioButton`

Widgets zur Eingabe der booleschen Werten.

### `geometry`

Geometrie-Widget. Wenn noch keine Geometrie im aktuellen Datensatz vorhanden ist, wird der Button "neue Geometrie" angezeigt, ansonsten der Button "Geometrie editieren". Die beiden Buttons schalten die visuelle Bearbeitungstools an.

### `measurement`

Widget für einen numerischen Wert mit einer Maßeinheit, wie z.B. `2 m`.

### `featureSelect`

Bei `M:1` Relationen, wie z.B. `relatedFeature`, zeigt eine Auswahlbox für das "parent" Feature an. Dieser Widget empfiehlt sich für "kleine" Referenztabellen (Layer) die im Klient komplett geladen werden (mit `loadingStrategy: all`).

Im nachfolgenden Beispiel sind für Straßenarbeiten (`roadwork`) vier Typen von Auftraggeber (`customer`) definiert.

@db_diagram
Straßenarbeiten und Auftraggeber.
@db_table "roadwork" "PK id|...|FK customer_type"
@db_table "customer" "Bund 1|Land 2|Kreis 3|Stadt 4"
@db_arrow_1 "roadwork:customer_type" "customer"
@end

Im Model `roadworkModel` kann man für `customer` diese Feldkonfiguration nutzen:

```
{
    name "customer"
    type "relatedFeature"
    title "Auftraggeber"
    relation {
        modelUid "customerModel"
    }
    foreignKey {
        name "customer_type"
    }
    widget {
        type "featureSelect"
    }
}
```

### `featureSuggest`

Bei `M:1` Relationen, zeigt eine Autocomplete-Box für das "parent" Feature mit der "live" Suche on Server-Seite. Diese Widget ist für größeren Tabellen (Layer), die nicht komplett geladen werden (`loadingStrategy: lazy`).

In diesem Beispiel sind Straßenarbeiten mit Straßen verbunden, wobei die Liste von Straßen serverseitig durchsucht wird:

@db_diagram
Straßenarbeiten und Straßen.
@db_table "roadwork" "PK id|...|FK street_id"
@db_table "street" "PK id|..."
@db_arrow_1 "roadwork:street_id" "street:id"
@end

```
{
    name "street"
    type "relatedFeature"
    title "Straße"
    relation {
        modelUid "streetModel"
    }
    foreignKey {
        name "street_id"
    }
    widget {
        type "featureSuggest"
    }
}
```





### `featureList`

Bei `1:M` Relationen, zeigt eine Liste der verknüpften Features mit entsprechenden Editierbuttons, die mit der Option `buttons` eingeschaltet werden können:

```
widget {
    type "featureList"
    buttons {
        add    true    // Feature hinzufügen
        edit   true    // Bearbeiten
        link   true    // Verknüpfen
        unlink true    // Abknüpfen
        delete true    // Löschen
    }
}
```

### `documentList`

Zeigt eine Liste der verknüpften Dokumenten. Ein "Dokument" ist ein Feature dessen Modell ein `file` bzw `blob` Field enthält. Neben den o.g. `button` Option, muss hier auch den Namen des File-Fields konfiguriert werden.

Angenommen es gibt zu Straßenarbeiten eine Liste von Dokumenten (`document`), wobei die Daten im Feld `datei` gespeichert werden.

@db_diagram
Straßenarbeiten und Dokumente.
@db_table "roadwork" "PK id|..."
@db_table "document" "PK id|...|FK roadwork_id|datei"
@db_arrow_m "document:roadwork_id" "roadwork:id"
@end

Dann kann man beide Modelle ungefähr wie folgt konfigurieren:

```
model {
    uid "documentModel"
    fields [
        { type "integer" name "id" isPrimaryKey true }
        { type "blob"    name "datei"  }
        {
            type "relatedFeature"
            name "roadwork"
            title "Maßnahme"
            relation {
                modelUid "roadworkModel"
            }
            foreignKey {
                name "roadwork_id"
            }
            widget {
                type "featureSuggest"
            }
        }
        ...
    ]
}

model {
    uid "roadworkModel"
    fields [
        { type "integer" name "id" isPrimaryKey true }
        {
            type "relatedFeatureList"
            name "documents"
            title "Dokumente für diese Maßnahme"
            widget {
                type "documentList"
                fileField {
                    name "datei"
                }
                buttons {
                    add  true
                    edit true
                }
            }
        }
        ...
    ]
}
```
