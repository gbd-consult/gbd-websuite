# Multifaktor-Adapter "totp" :/admin-de/konfiguration/authMultiFactorAdapter/totp

Der Adapter `totp` fügt als zweiten Faktor ein zeitbasiertes Einmalkennwort (TOTP) hinzu, das der Nutzer in einer Authenticator-App erzeugt. Der Nutzer muss dazu ein hinterlegtes Geheimnis (`mfaSecret`) besitzen. Die Erzeugung und Prüfung der Kennwörter steuern Sie bei Bedarf über die OTP-Optionen `otp`.

## Beispiel-Konfiguration ::

```javascript
auth.mfa+ {
    uid "AUTH_MFA_TOTP"
    type "totp"
    message "Bitte geben Sie den Code aus Ihrer Authenticator-App ein."
}
```

Der Adapter wird über `auth.mfa+` aktiviert. Die `uid` referenziert der jeweilige Nutzer über sein Attribut `mfaUid`, um diesen zweiten Faktor zu verwenden; zusätzlich muss der Nutzer ein Geheimnis (`mfaSecret`) besitzen. `message` ist der Hinweistext, der bei der Code-Eingabe angezeigt wird. Erzeugung und Prüfung der Kennwörter passen Sie bei Bedarf über `otp` an.

%ref "gws.plugin.auth_mfa.totp.Config"
