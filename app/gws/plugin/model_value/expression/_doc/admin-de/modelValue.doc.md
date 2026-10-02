# Modell-Wert "expression" :/admin-de/konfiguration/modelValue/expression

Die Wertquelle `expression` ermittelt den Feldwert, indem sie einen Python-Ausdruck auswertet. Im Ausdruck stehen unter anderem die aktuelle Anwendung, der Nutzer, das Projekt und das bearbeitete Objekt zur Verfügung. Den Ausdruck geben Sie über `expression` an; zusätzliche Module machen Sie mit `imports` verfügbar.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "flaeche"
    type "float"
    title "Fläche"
    values+ {
        type "expression"
        expression "feature.get('breite') * feature.get('hoehe')"
    }
}
```

Die Wertquelle `expression` berechnet den Feldwert aus einem Python-Ausdruck. Über `feature.get(...)` greifen Sie auf andere Attribute des bearbeiteten Objekts zu, hier auf `breite` und `hoehe`.

%ref "gws.plugin.model_value.expression.Config"
