# Modell "default" :/admin-de/konfiguration/model/default

Der Modell-Typ `default` ist das allgemeine Standardmodell ohne eigene Datenquelle. Er übernimmt alle Attribute der Features unverändert, unabhängig von den konfigurierten `fields`, und kommt dort zum Einsatz, wo kein Modell konfiguriert ist. Die Features werden vollständig geladen; als Kennung und Geometriefeld dienen fest `uid` und `geometry`.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "default"
    fields+ {
        name "uid"
        type "integer"
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

Das `default`-Modell besitzt keine eigene Datenquelle und beschreibt die Features allein über die konfigurierten `fields`. Das Feld `uid` hält die Kennung, `name` ein Textattribut und `geometry` die Geometrie. Kennung und Geometriefeld sind beim `default`-Modell fest `uid` und `geometry`; `isPrimaryKey` an einem anderen Feld ändert daran nichts.

%ref "gws.base.model.default_model.Config"
