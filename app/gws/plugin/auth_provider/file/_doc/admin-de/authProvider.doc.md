# Authentifizierungs-Provider "file" :/admin-de/konfiguration/authProvider/file

Der Provider `file` liest Benutzerkonten aus einer JSON-Datei, deren Pfad Sie über `path` angeben. Jeder Datensatz enthält Login und ein als Hash hinterlegtes Passwort, weitere Felder werden zu Nutzereigenschaften. Er eignet sich für kleine, überschaubare Nutzerbestände ohne externen Verzeichnisdienst.

## Beispiel-Konfiguration ::

```javascript
auth.providers+ {
    type "file"
    path "/data/auth/users.json"
}
```

Der Provider wird über `auth.providers+` aktiviert. `path` verweist auf die JSON-Datei mit den Benutzerkonten. Jeder Datensatz enthält mindestens `login` und ein als Hash hinterlegtes `password`; weitere Felder wie `displayName` oder `roles` werden zu Eigenschaften des Nutzers. Den Passwort-Hash erzeugen Sie mit dem CLI-Befehl `gws auth password`.

%ref "gws.plugin.auth_provider.file.Config"
