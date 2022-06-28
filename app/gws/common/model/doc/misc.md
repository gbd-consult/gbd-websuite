## Felder Flags

Flags (booleschen Optionen) für Felder:

- `isPrimaryKey`: das Feld ist ein Primärschlüssel, oder ein Teil davon
- `isSearchable`: das Feld kann für die Volltextsuche verwendet werden

## Zugriffsrechte für Felder


Für jedes Feld können `read` (lesen) und `write` (schreiben) Zugriffsrechte definiert werden. Im folgenden Beispiel ist das Feld "Adresse" von den Gruppen "mitarbeiter" und "editor" lesbar, aber nur von "editor" schreibbar:

```
{
    name "address"
    type "string"
    title "Adresse"
    read {
        access [
            { role "mitarbeiter" type allow }
            { role "editor"      type allow }
            { role "all"         type deny  }
        ]
    }
    write {
        access [
            { role "editor"      type allow }
            { role "all"         type deny  }
        ]
    }
}
```

## Default-Werte


-- TODO

## Validierungen

-- TODO

## Widgets

-- TODO

