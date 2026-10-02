# Modell-Feld "float" :/admin-de/konfiguration/modelField/float

Das Feld `float` bildet ein Attribut mit einem Gleitkommawert ab. Sie verwenden es für numerische Angaben mit Nachkommastellen, etwa Messwerte oder Beträge.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "flaeche"
    type "float"
    title "Fläche (m²)"
    widget { type "float" }
}
```

Das Feld `flaeche` speichert einen Gleitkommawert. Das Widget `float` ist die Vorgabe für `float`-Felder und akzeptiert Nachkommastellen.

%ref "gws.plugin.model_field.float.Config"
