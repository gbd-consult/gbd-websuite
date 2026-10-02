# Authentifizierungs-Provider "postgres" :/admin-de/konfiguration/authProvider/postgres

Der Provider `postgres` prüft Zugangsdaten über SQL-Abfragen gegen eine PostgreSQL-Datenbank. Mit `authorizationSql` legen Sie die Anmelde-Abfrage fest, mit `getUserSql` die Abfrage eines Nutzerdatensatzes anhand seiner ID; die Datenbankverbindung wählen Sie über `dbUid`.

Beide Abfragen erhalten ihre Parameter als Platzhalter in doppelten geschweiften Klammern: `authorizationSql` bekommt `{{username}}`, `{{password}}` und/oder `{{token}}` (je nach Methode), `getUserSql` bekommt `{{uid}}`.

Die Anmelde-Abfrage muss **genau eine Zeile** mit mindestens diesen Spalten liefern (Groß-/Kleinschreibung egal); liefert sie keine Zeile, wird der nächste Provider versucht:

| Spalte | Bedeutung |
|---|---|
| `validuser` | `true`, wenn der Nutzer sich anmelden darf |
| `validpassword` | `true`, wenn das Passwort stimmt |
| `uid` | eindeutige Benutzer-Kennung |
| `roles` | kommagetrennte Liste der Rollen |

Weitere Spalten werden zu Eigenschaften des Nutzers (etwa `displayName`, `login`). Das Passwort prüfen Sie in der Abfrage selbst, üblicherweise mit `pgcrypto`:

```sql
( passwd = crypt({{password}}, passwd) ) AS validpassword
```

## Beispiel-Konfiguration ::

```javascript
auth.providers+ {
    type "postgres"
    dbUid "main_db"
    authorizationSql '''
        SELECT
            id                              AS uid,
            login                           AS login,
            enabled                         AS validuser,
            (passwd = crypt({{password}}, passwd)) AS validpassword,
            roles                           AS roles
        FROM public.users
        WHERE login = {{username}}
    '''
    getUserSql '''
        SELECT id AS uid, login AS login, roles AS roles
        FROM public.users
        WHERE id = {{uid}}
    '''
}
```

Der Provider wird über `auth.providers+` aktiviert und nutzt über `dbUid` den angegebenen Datenbank-Provider. `authorizationSql` erhält `{{username}}` und `{{password}}` und muss genau eine Zeile mit den Spalten `uid`, `validuser`, `validpassword` und `roles` liefern; die Passwortprüfung erfolgt hier mit `pgcrypto`. `getUserSql` liefert zu einer `{{uid}}` den Nutzerdatensatz für eine bestehende Sitzung.

%ref "gws.plugin.postgres.auth_provider.Config"
