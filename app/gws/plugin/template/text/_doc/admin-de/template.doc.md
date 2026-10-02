# Vorlage "text" :/admin-de/konfiguration/template/text

Die Vorlage `text` erzeugt reine Textausgaben. Sie beruht wie die HTML-Vorlage auf der Jump-Sprache, unterstützt aber keine Sonderbefehle für Karte oder Legende und keine Nicht-Text-Ausgaben. Den Vorlageninhalt geben Sie über `text` oder als Datei über `path` an.

## Beispiel-Konfiguration ::

```javascript
templates+ {
    subject "feature.title"
    type "text"
    text "{{{{vorname}}}} {{{{nachname}}}}"
}
```

Das `subject` bestimmt den Zweck der Vorlage, hier den Titel eines Features. Die Platzhalter `{{{{vorname}}}}` und `{{{{nachname}}}}` werden durch die Feldwerte des Features ersetzt. Da die Ausgabe reiner Text ist, entfallen HTML-Auszeichnung und die Sonderbefehle der HTML-Vorlage.

%info
Die Konfiguration wird selbst mit der Jump-Sprache verarbeitet. Eine `{`, auf die kein Leerzeichen folgt, deutet der Server daher schon beim Einlesen der Konfiguration als Beginn eines Ausdrucks. Eine über `text` eingebettete Vorlage enthält aber ihrerseits Jump-Platzhalter wie `{{vorname}}`. Damit diese nicht bereits beim Einlesen ausgewertet werden, verdoppeln Sie die Klammern: `{{{{vorname}}}}` wird beim Einlesen zu `{{vorname}}`, das die Vorlage anschließend gegen das Feature auflöst. Geben Sie die Vorlage über `path` als eigene Datei an, entfällt das Verdoppeln – die Datei durchläuft die Konfigurationsverarbeitung nicht und verwendet einfache Klammern (`{{vorname}}`).
%end

%ref "gws.plugin.template.text.Config"
