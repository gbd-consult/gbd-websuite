# Modell-Feld "bool" :/admin-de/konfiguration/modelField/bool

Das Feld `bool` bildet ein Attribut mit einem Wahrheitswert (wahr oder falsch) ab. Sie verwenden es für Ja/Nein-Angaben; im Editor wird der Wert standardmäßig über einen Umschalter dargestellt.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "aktiv"
    type "bool"
    title "Aktiv"
    widget { type "toggle" }
}
```

Das Feld `aktiv` speichert einen Wahrheitswert. Ohne eigene Angabe erzeugt die WebSuite für `bool`-Felder automatisch das Widget `toggle`, hier ist es zur Verdeutlichung ausgeschrieben.

%ref "gws.plugin.model_field.bool.Config"
