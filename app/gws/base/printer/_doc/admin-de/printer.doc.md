# Drucker :/admin-de/konfiguration/printer

Ein Drucker erzeugt beim [](/admin-de/themen/darstellung/drucken) aus der aktuellen Kartenansicht ein PDF. Sie konfigurieren ihn global oder je Projekt; ein Projekt kann mehrere Drucker für unterschiedliche Vorlagen enthalten.

Mit `template` geben Sie die zu verwendende Druckvorlage an. Über `qualityLevels` definieren Sie die auswählbaren Qualitätsstufen mit ihrer jeweiligen Auflösung (DPI); mit `models` binden Sie Datenmodelle für bedruckbare Objekte ein.

## Beispiel-Konfiguration ::

```javascript
printers+ {
    title "A4 Querformat"
    template {
        type "html"
        path "/data/print/a4-quer.cx.html"
        mapSize ["270mm" "170mm"]
    }
    qualityLevels [
        { name "Entwurf" dpi 72 }
        { name "Druck" dpi 300 }
    ]
}
```

`template` verweist auf die Druckvorlage – hier eine [`html`-Vorlage](/admin-de/konfiguration/template/html) mit den Druckbefehlen `@page`, `@map` und `@legend`. `mapSize` legt die Größe des Kartenbildes in der Vorlage fest. Die beiden `qualityLevels` bieten dem Nutzer im Druckdialog die Auflösungen „Entwurf" (72 DPI, schnell) und „Druck" (300 DPI, hochauflösend) zur Wahl; `name` erscheint in der Auswahl, `dpi` bestimmt die Rasterauflösung.

%ref "gws.base.printer.core.Config"
%demo "print_fields"
%demo "print_multi_page"
%demo "print_simple"
%demo "print_templates"
%demo "print_vector"
