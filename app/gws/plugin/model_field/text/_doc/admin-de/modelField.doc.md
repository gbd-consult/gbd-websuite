# Modell-Feld "text" :/admin-de/konfiguration/modelField/text

Das Feld `text` bildet ein Attribut mit einem Zeichenketten-Wert ab. Über die Option `textSearch` legen Sie fest, ob und wie das Feld bei der stichwortbasierten Suche berücksichtigt wird, etwa mit welchem Suchtyp und ob die Groß- und Kleinschreibung beachtet wird.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "name"
    type "text"
    title "Name"
    widget { type "input" }
    textSearch { type "any" minLength 3 }
}
```

Das Feld bildet die Spalte `name` ab. `widget` bestimmt das Eingabeelement im Client – hier ein einfaches Textfeld (`input`). `textSearch` macht das Feld für die stichwortbasierte Suche zugänglich: `type "any"` findet jeden Teilstring, `minLength 3` verlangt mindestens drei Zeichen, bevor gesucht wird.

%ref "gws.plugin.model_field.text.Config"
