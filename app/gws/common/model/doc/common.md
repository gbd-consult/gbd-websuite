## Zugriffsrechte

Für Modelle können Zugriffsrechte definiert werden.  Es werden folgende Rechte unterstützt:


Bezeichnung|Bedeutung
---|---
`create` | det Nutzer kann neue Records anlegen
`delete` | det Nutzer kann Records löschen
`read` | det Nutzer kann Felder lesen
`write` | det Nutzer kann Felder aktualisieren



Für jedes Feld können zusätzlich `read` (lesen) und `write` (aktualisieren) Zugriffsrechte definiert werden.

```
{
    uid "person_model"

    permissions {
        // "mitarbeiter" und "editor" können lesen
        read {
            access [
                { role "mitarbeiter" type allow }
                { role "editor"      type allow }
                { role "all"         type deny  }
            ]
        }
        // nur "editor" können Felder updaten
        write {
            access [
                { role "editor"      type allow }
                { role "all"         type deny  }
            ]
        }
        // nur "editor" können neue Records erstellen
        create {
            access [
                { role "editor"      type allow }
                { role "all"         type deny  }
            ]
        }
        // nur "it" kann Records löschen
        delete {
            access [
                { role "it"          type allow }
                { role "all"         type deny  }
            ]
        }
    ]

    fields [
        ...
        {
            name "address"
            type "string"
            title "Adresse"

            permissions {
                // "adresse" in nur von "ema" lesbar
                read {
                    access [
                        { role "ema"   type allow }
                        { role "all"   type deny  }
                    ]
                }
                // "adresse" in nur von "ema" schreibbar
                write {
                    access [
                        { role "ema"   type allow }
                        { role "all"   type deny  }
                    ]
                }
            }
        }
        ...
    ]
}
```

## Felder allgemein

Die Konfiguration eines Feldes ist umfangreich und beinhaltet in der Regel:

- allgemeine Attribute wie Name, Titel und Typ
- Daten- und Relations- Attribute
- Default Werte
- mehrere Validierungsregel
- eine Formularfeld (`widget`) Konfiguration

Allgemeine Optionen für Felder sind:


Option | Typ | Bedeutung
-------|-----|------
`typ` |  `string`| Feldtyp
`name`  | `string` | Name (interne Bezeichnung)
`title` | `string`| Beschriftung
`isPrimaryKey` |  `boolean`| das Feld ist ein Primärschlüssel, oder ein Teil davon
`isSearchable` |  `boolean`| das Feld kann für die Volltextsuche verwendet werden



