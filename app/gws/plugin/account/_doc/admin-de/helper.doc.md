# Helfer "account" :/admin-de/konfiguration/helper/account

Der Hilfsdienst `account` verwaltet Benutzerkonten in einer Datenbanktabelle: die administrative Pflege, das E-Mail-gestützte Onboarding und optional die Multifaktor-Authentifizierung. Die Tabelle muss die festen Spalten `email`, `status`, `password`, `mfauid`, `mfasecret`, `tc`, `tctime` und `tccategory` enthalten; die fachliche Einordnung und der Ablauf stehen im Thema [Benutzerkonten](/admin-de/themen/zugriff/konten).

Das Datenmodell für die administrative Bearbeitung geben Sie über `adminModel` an, ein optionales `userModel` erlaubt Nutzern die Pflege eigener Stammdaten. Mit `onboardingUrl` legen Sie die Adresse fest, an die Freischaltungs-Mails verweisen, mit `tcLifeTime` die Gültigkeit des Freischaltungs-Codes (Vorgabe eine Stunde); über `mfa` bestimmen Sie die wählbaren Verfahren zur Multifaktor-Authentifizierung. Der Mail-Versand nutzt den konfigurierten E-Mail-Helfer.

## Beispiel-Konfiguration ::

```javascript
helpers+ {
    type "account"
    usernameColumn "login"
    onboardingUrl "https://gws.stadt.example/project/user_account"

    mfa [
        { mfaUid ""              title "keine Multi-Faktor-Authentisierung" }
        { mfaUid "AUTH_MFA_TOTP" title "per Authenticator-App" }
    ]

    adminModel {
        type "postgres"
        tableName "public.account"
        isEditable true
        permissions.edit "allow admin, deny all"
        fields+ { name "id" type "integer" isPrimaryKey true }
        fields+ { name "login" type "text" }
        fields+ { name "email" type "text" }
        fields+ { name "status" type "integer" }
    }

    templates+ {
        subject "onboarding.emailBody"
        type "text"
        text "Klicken Sie auf {{url}}, um Ihr Konto zu aktivieren."
    }
}
```

`adminModel` beschreibt das Datenmodell, mit dem Administratoren die Konten in der Tabelle `public.account` pflegen; neben den hier aufgeführten Feldern muss die Tabelle die festen Spalten für Passwort, Status und Freischaltung enthalten. `usernameColumn "login"` legt fest, welche Spalte als Anmeldename dient. `onboardingUrl` ist das Ziel der Freischaltungs-Mails, deren Text die Vorlage mit dem Subject `onboarding.emailBody` liefert. `mfa` bietet dem Nutzer die aufgeführten Verfahren zur Multifaktor-Authentifizierung zur Auswahl an. Für die tatsächliche Anmeldung ergänzen Sie den Auth-Provider `account`.

%ref "gws.plugin.account.helper.Config"
