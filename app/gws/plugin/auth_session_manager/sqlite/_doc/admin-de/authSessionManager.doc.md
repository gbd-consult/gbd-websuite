# Sitzungsverwaltung "sqlite" :/admin-de/konfiguration/authSessionManager/sqlite

Der Sitzungsverwalter `sqlite` speichert die Sitzungen in einer SQLite-Datenbank, sodass sie einen Neustart des Servers überdauern. Den Speicherort der Datenbankdatei legen Sie über `path` fest; ohne Angabe wird ein versionsabhängiger Standardpfad verwendet.

## Beispiel-Konfiguration ::

```javascript
auth.session {
    type "sqlite"
    lifeTime "3600"
    path "/data/sessions.sqlite"
}
```

Die Sitzungsverwaltung wird über `auth.session` gesetzt (ein einzelnes Objekt, keine Liste). `lifeTime` gibt die Lebensdauer einer Sitzung in Sekunden an, hier eine Stunde. `path` bestimmt den Speicherort der SQLite-Datei; ohne Angabe wird ein versionsabhängiger Standardpfad im internen Verzeichnis verwendet. Wird keine Sitzungsverwaltung konfiguriert, kommt automatisch `sqlite` zum Einsatz.

%ref "gws.plugin.auth_session_manager.sqlite.Config"
