# Aktion "web" :/admin-de/konfiguration/action/web

Die Aktion `web` liefert dynamische Assets aus, also Dateien aus dem globalen oder projektbezogenen `assets`-Verzeichnis. Passende Vorlagen werden dabei zur Laufzeit gerendert, andere Dateien nach Prüfung der MIME-Filter direkt ausgegeben. Sie stellt außerdem Endpunkte für Downloads und den Zugriff auf Objekt-Dateien bereit.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "web"
}
```

Die Aktion selbst hat keine eigenen Optionen; das Aktivieren schaltet die Asset-, Download- und Objekt-Datei-Endpunkte frei. Welche Verzeichnisse ausgeliefert und welche URLs auf die Endpunkte umgeschrieben werden, konfigurieren Sie getrennt über `web.sites` (`assets.dir`, `root.dir`, `rewriteRules`).

%ref "gws.base.web.action.Config"
