# Aktion "printer" :/admin-de/konfiguration/action/printer

Die Aktion `printer` stellt die Druckschnittstelle bereit. Sie startet Druckaufträge im Hintergrund, überwacht deren Status und liefert das erzeugte Dokument zum Download aus.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "printer"
}

printers+ {
    template {
        type "html"
        path "print.cx.html"
        mapSize [ "150mm" "100mm" ]
    }
    qualityLevels [
        { dpi 150 name "150 dpi" }
    ]
}
```

Die Aktion selbst hat keine eigenen Optionen; sie stellt nur die Druckschnittstelle bereit. Die tatsächlich verfügbaren Druckvorlagen konfigurieren Sie getrennt als `printers`: `template` bestimmt Vorlage und Kartenformat (`mapSize`), `qualityLevels` bietet dem Nutzer wählbare Auflösungen an.

%ref "gws.base.printer.action.Config"
