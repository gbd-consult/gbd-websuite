## Skalare Felder

Skalare Felder repräsentieren Datenbank Spalten und Feature Attribute.

### Primitive Datentypen

Primitive Datentypen entsprechen direkt den Datenbank-Typen. Es werden folgende Datentypen unterstützt:

- `string`
- `integer`
- `float`
- `boolean`
- `date`
- `datetime`

### Komplexe Datentypen

Komplexe Datentypen werden als primitive Datenbank-Typen gespeichert, haben jedoch eine andere Semantik. Zu komplexen Datentypen gehören

- `measurement`: numerischer Wert mit einer Maßeinheit, wie `2 m` oder `4 ha`
- `money`: monetärer Wert wie `12.34`
- `currency`: monetärer Wert mit Währung, wie `12.34 EUR`
- `crs` KBS Wert wie `3857`
- `extent`: 4 reelle Zahlen die eine räumliche BoundingBox beschreiben

### Datei

Für Dateien sind 2 Feldtypen vorhanden:

- `file`: die Datei wird in Dateisystem abgelegt. Mit der Option `path` kann man eine Vorlage für Dateipfad konfigurieren. Im Feld selbst wird nur dieser Pfad gespeichert.

- `blob`: die Datei wird als `BLOB` direkt in der DB abgelegt

@quote
```
{
    name "id"
    type "integer"
    isPrimaryKey true
}
{
    name "attachment"
    type "file"
    title "Anlage"
    // in "path" können andere Felder des Models verwendet werden,
    // sowie "{file.name}" und "{file.extension}" für Original-Namen und Erweiterung
    filePath "/extern/files/emails/{id}.{file.extension}"
}
{
    name "avatar"
    type "blob"
    title "Avatar"
}
```
@end


### Geometrie

Geometrien werden als Feldtyp `geometry` konfiguriert. Einen konkreten Geometrietyp mit KBS beschreibt man mit der Option `geometry`:

```
{
    name "geom"
    type "geometry"
    geometry { type "LineString" crs 3857 }
}
```

### Fest- und Defaultwerte

Mit der Option `value` kann man für skalare Typen Fest- und Defaultwerte definieren.

Feste Werte nutzt man, falls die Daten aus der Quelle (`read`) oder dem Formular (`write`) ignoriert werden müssen und stattdessen ein vordefinierter Wert verwendet. Defaultwerte verwendet man, wenn der entsprechende Wert nicht vorhanden ist.

Für die `value` Option können folgende Eigenschaften definiert werden:

Typ | Bedeutung
---|---
`read` | beim Lesen aus der Quelle wird der angegebene Wert verwendet
`write` | beim Schreiben in die Quelle wird der angegebene Wert verwendet
`readDefault` | beim Lesen wird der angegebene Wert verwendet falls der Quellenwert `NULL` oder nicht vorhanden ist. Beachten Sie, dass ein leeres String in diesem Fall ein legitimer Wert ist.
`writeDefault` |  beim Schreiben in die Quelle werden `NULL` Werte und leere Strings mit diesem Wert ersetzt


Zu diesen Eigenschaften werden Value Objekte verknüpft, die unterschiedliche Typen haben können:

Typ | Bedeutung
---|---
`static` | ein literaler Wert
`expression` | ein Python Ausdruck
`template` | eine Vorlage

@quote
```
model {
    ...
    fields [

        // falls "status" NULL ist, "Unbekannt" im Formular zeigen
        {
            name "status"
            type "string"
            value {
                readDefault {
                    type "static"
                    value "Unbekannt"
                }
            }
        }

        // falls "anzahl" leer ist, 0 in die DB schreiben
        {
            name "anzahl"
            type "integer"
            value {
                writeDefault  {
                    type "static"
                    value 0
                }
            }
        }

        // in "time_updated" stets die aktuelle Zeit schreiben
        {
            name "time_updated"
            type "datetime"
            value {
                write {
                    type "expression"
                    text "datetime.datetime.now()"
                }
            }
        }

        // das Feld "Vollname" wird beim Lesen aus "Vorname" und "Name"
        // mit einer Template zusammengesetzt
        {
            name "full_name"
            type "string"
            value {
                read {
                    type "template"
                    template {
                        type "format"
                        text "{vorname} {name}"
                    }
                }
            }
        }
        ...
    ]
}
```
@end


### Volltextsuche

Skalare Felder können für die Volltextsuche im Client verwendet werden. Dafür muss eine `textSearch` Konfiguration vorhanden sein, mit folgenden Eigenschaften:

Typ | Bedeutung
---|---
`type` | Suchmodus
`caseSensitive` | `true` falls die Suche case-sensitiv erfolgt, default `false`
`minLength` | min. Länge des Suchbegriffs, default `1`

Für `type` sind folgende Werte möglich:

Wert | Bedeutung
---|---
`exact` | das Feld ist gleich dem Suchbegriff
`like` | das Feld enthält den Suchbegriff
`begin` | das Feld beginnt mit dem Suchbegriff
`end` | das Feld endet mit dem Suchbegriff

Im folgenden Beispiel ist für `surname` die `like` Suche konfiguriert, mit mindestens 3 Zeichen, und das Feld `code` muss exact übereinstimmen:


@quote
```
model {
    ...
    fields [

        {
            name "surname"
            type "string"
            title "Nachname"
            textSearch {
                type "like"
                minLength 3
            }
        }
        {
            name "code"
            type "string"
            title "Artikelnummer"
            textSearch {
                type "exact"
                caseSensitive true
            }
        }
```
@end



