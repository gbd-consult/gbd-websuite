## Validatoren

An einem Modell-Feld kann eine beliebige Anzahl von Validatoren angehängt werden. Beim Erstellen oder Aktualisieren von Features werden alle Validatoren nach der Reihe überprüft. Falls einer der Validatoren fehlschlägt, wird das Aktualisieren abgebrochen und dem Nutzer eine entsprechende Fehlermeldung angezeigt.

Für einen Validator muss zumindest der Typ (`type`) und die Fehlermeldung (`messsage`) konfiguriert werden. Manche Validatoren haben auch weitere Optionen.

Ein Feld mit Validatoren kann wie folgt aussehen:

```
{
    name "code"
    type "integer"
    title "Zugangscode"

    validators [
        // das ist ein Pflichtfeld
        {
            type     "required"
            message  "Bitte Zugangscode eingeben"

        }
        // Der Wert muss zwischen 1000 und 9999 sein
        {
            type      "range"
            minValue  { type "static" value 1000 }
            maxValue  { type "static" value 9999 }
            message   "Zugangscode muss genau 4 Zahlen sein"
        }
        // ...darf jedoch nur mit 2, 3 or 9 beginnen
        {
            type      "pattern"
            pattern   "^[239]"
            message   "falscher Zugangscode"
        }
    ]
}
```

Folgende Validatortypen sind definiert:

### `required`

Das Feld darf nicht leer (also `NULL` oder ein leerer String) sein.

### `pattern`

Der Stringwert des Feldes muss mit einem regulären Ausdruck übereinstimmen.

```
{
    name "email"
    type "string"
    validators [
        // z.B. "max.mustermann@firma.de"
        {
            type     "pattern"
            pattern  "^[a-z]+.[a-z]@firma\.de"
        }
    ]
}
```

### `length`

Die Länge des Wertes muss zwischen `minLength` und `maxLength` (inklusive) liegen. Bei Relationen wird die Anzahl der verknüpften Feature geprüft. Es kann nur "min" oder nur "max" konfiguriert werden, oder beides.

```
{
    name "images"
    type "relatedFeatureList"
    validators [
        // von 1 bis 5 Bilder zulässig
        {
            type       "length"
            minLength  1
            maxLength  5
        }
    ]
}
```

### `range`

Der Wert muss zwischen `minValue` und `maxValue` liegen. Es können `static` oder `expression` Value Objekte verwendet werden.

```
{
    name "end_date"
    type "date"
    title "Enddatum"

    validators [
        // "Enddatum" muss nach dem "Beginndatum" sein
        {
            type "range"
            minValue {
                type "expression"
                text "begin_date"
            }
            message "Enddatum ist früher als Beginndatum"
        }

        // Enddatum muss bevor 1. Jan 2050 liegen
        {
            type "range"
            maxValue {
                type  "static"
                value "2050-01-01"
            }
            message "Enddatum ist viel zu weit in der Zukunft"
        }
    ]
}
```

### `geometry`

Führt eine räumliche Validierung aus. Die Validierungsoperation kann `intersects`, `within` oder `disjoint` sein, als Wert können Extents (vier Koordinaten), WKT oder GeoJSON verwendet werden.

```
{
    name "geom"
    type "geometry"
    geometry { type "point" crs 3857 }

    validators [
        // muss innerhalb dieses Polygons liegen
        {
            type       "geometry"
            operation  "within"
            value {
                type "static"
                value "POLYGON((0 0) (1 1) (2 2) (0 0))"
            }
            message "falsche Koordinaten!"
        }
    ]
}
```
