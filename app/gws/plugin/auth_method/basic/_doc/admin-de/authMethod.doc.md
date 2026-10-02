# Authentifizierungsmethode "basic" :/admin-de/konfiguration/authMethod/basic

Die Methode `basic` übermittelt die Zugangsdaten per HTTP-Basic-Authentifizierung. Der Browser sendet Benutzername und Passwort im `Authorization`-Header, die dann an die Provider weitergereicht werden. Über `realm` legen Sie die Kennung des Anmeldebereichs fest, die der Browser im Anmeldedialog anzeigt.

## Beispiel-Konfiguration ::

```javascript
auth.methods+ {
    type "basic"
    realm "GBD WebSuite"
}
```

Die Methode wird über `auth.methods+` aktiviert. `realm` legt die Kennung fest, die der Browser im Anmeldedialog anzeigt. Standardmäßig ist die Methode nur über HTTPS nutzbar (`secure`); für den Zugriff über unverschlüsselte interne Netze können Sie mit `allowInsecureFrom` einzelne IP-Adressen freigeben.

%ref "gws.plugin.auth_method.basic.Config"
