# Authentifizierungs-Provider "account" :/admin-de/konfiguration/authProvider/account

Der Provider `account` prüft Zugangsdaten gegen die vom Konten-Modul verwalteten Benutzerkonten. Nur aktive Konten werden zur Anmeldung zugelassen, Nutzer und Rollen ergeben sich aus dem jeweiligen Konten-Datensatz. Er benötigt keine eigenen Optionen, setzt aber den konfigurierten `account`-Helper voraus.

## Beispiel-Konfiguration ::

```javascript
auth.providers+ {
    type "account"
}
```

Der Provider selbst hat keine Optionen und wird allein über `auth.providers+` aktiviert. Die Konten – Tabelle, Anmeldespalte, Freischaltung und Multifaktor – beschreiben Sie im [`account`-Helper](/admin-de/konfiguration/helper/account), den dieser Provider voraussetzt. Bei der Anmeldung lässt der Provider nur Konten mit aktivem `status` zu; Nutzer und Rollen ergeben sich aus dem jeweiligen Datensatz.

%ref "gws.plugin.account.auth_provider.Config"
