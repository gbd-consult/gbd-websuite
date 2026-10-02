# Aktion "project" :/admin-de/konfiguration/action/project

Die Aktion `project` liefert dem Client die Konfiguration und Metadaten eines Projekts sowie die Angaben zu Locale und angemeldetem Nutzer. Sie muss aktiv sein, damit ein Projekt im Client geladen werden kann.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "project"
}
```

Die Aktion hat keine eigenen Optionen; das Aktivieren genügt. Sie können sie global (für alle Projekte) oder je Projekt eintragen. Mit `access` bzw. `permissions` beschränken Sie, welche Nutzer das Projekt laden dürfen.

%ref "gws.base.project.action.Config"
