# Authentifizierungs-Provider "ldap" :/admin-de/konfiguration/authProvider/ldap

Der Provider `ldap` prüft Zugangsdaten gegen einen LDAP- oder ActiveDirectory-Verzeichnisdienst. Den Server geben Sie als LDAP-URL über `url` an; mit `bindDN` und `bindPassword` hinterlegen Sie ein Konto mit Suchberechtigung. Über `users` bilden Sie LDAP-Filter oder Gruppenzugehörigkeiten auf GBD-WebSuite-Rollen ab.

## Beispiel-Konfiguration ::

```javascript
auth.providers+ {
    type "ldap"
    url "ldap://ldap.example.org:389/dc=example,dc=org?uid"
    bindDN "cn=admin,dc=example,dc=org"
    bindPassword "secret"
    users+ {
        memberOf "(cn=gis-editors)"
        roles [ "editor" ]
    }
    users+ {
        matches "(objectClass=person)"
        roles [ "user" ]
    }
}
```

Der Provider wird über `auth.providers+` aktiviert. `url` folgt dem Schema `ldap://host:port/baseDN?searchAttribute`; das Attribut am Ende (hier `uid`) wird zum Abgleich des Logins verwendet. `bindDN` und `bindPassword` hinterlegen ein Konto mit Suchberechtigung. Jeder `users`-Eintrag bildet eine Bedingung auf GBD-WebSuite-Rollen ab: `memberOf` ist ein LDAP-Filter für Gruppen und prüft, ob der Nutzer in `member`, `members` oder `uniqueMember` einer passenden Gruppe steht; `matches` ist ein beliebiger LDAP-Filter, auf den der Eintrag des Nutzers passen muss. Ein Nutzer erhält die `roles` aller zutreffenden Einträge.

%ref "gws.plugin.auth_provider.ldap.Config"
