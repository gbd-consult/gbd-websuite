# Modell "default" :/admin-de/konfiguration/model/default

Der Modell-Typ `default` ist das allgemeine Standardmodell ohne eigene Datenquelle. Er verwaltet Features anhand der von Ihnen konfigurierten `fields` und kommt überall dort zum Einsatz, wo kein quellenspezifischer Modell-Typ angegeben ist. Die Features werden vollständig in den Speicher geladen; als Standard dienen `uid` und `geometry` als Kennung und Geometriefeld.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "default"
    fields+ {
        name "id"
        type "integer"
        isPrimaryKey true
    }
    fields+ {
        name "name"
        type "text"
    }
    fields+ {
        name "geometry"
        type "geometry"
    }
}
```

Das `default`-Modell besitzt keine eigene Datenquelle und beschreibt die Features allein über die konfigurierten `fields`. Das Feld `id` dient mit `isPrimaryKey true` als Kennung, `name` hält ein Textattribut und `geometry` die Geometrie. Ohne abweichende Angabe werden `uid` und `geometry` als Standard-Kennung bzw. -Geometriefeld verwendet.

%ref "gws.base.model.default_model.Config"
