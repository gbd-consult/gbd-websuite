# Modell-Wert "currentUser" :/admin-de/konfiguration/modelValue/currentUser

Die Wertquelle `currentUser` setzt den Feldwert anhand des aktuell angemeldeten Nutzers. Ohne weitere Angabe wird der Anmeldename (`loginName`) eingetragen. Über `format` geben Sie eine Formatvorlage an, die auf Eigenschaften des Nutzers zugreift, etwa `"{user.displayName}"`.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "updated_by"
    type "text"
    title "Bearbeitet von"
    permissions.edit "deny all"
    values+ {
        type "currentUser"
        forRead false
        forUpdate true
    }
}
```

Die Wertquelle `currentUser` trägt den Anmeldenamen des aktuellen Nutzers ein. Mit `forUpdate true` und `forRead false` wird der Wert nur beim Speichern gesetzt, nicht beim Lesen; `permissions.edit "deny all"` schützt das Feld vor manueller Änderung.

%ref "gws.plugin.model_value.current_user.Config"
%demo "value_current_user"
