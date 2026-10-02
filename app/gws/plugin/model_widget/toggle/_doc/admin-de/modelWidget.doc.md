# Modell-Widget "toggle" :/admin-de/konfiguration/modelWidget/toggle

Das Widget `toggle` stellt ein Umschaltelement dar und eignet sich für Felder vom Typ `bool`. Mit der Option `kind` legen Sie fest, ob das Element als Kontrollkästchen (`checkbox`) oder als Optionsschalter (`radio`) angezeigt wird.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "aktiv"
    type "bool"
    title "Aktiv"
    widget {
        type "toggle"
        kind "checkbox"
    }
}
```

Das `bool`-Feld `aktiv` wird als Umschaltelement dargestellt. Mit `kind "checkbox"` erscheint es als Kontrollkästchen; alternativ zeigt `kind "radio"` einen Optionsschalter.

%ref "gws.plugin.model_widget.toggle.Config"
