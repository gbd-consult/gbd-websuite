# Authentifizierung :/admin-de/themen/zugriff/authentifizierung

*Authentifizierung* stellt die Identität eines Nutzers fest. Zunächst ist jeder Nutzer anonym. Über eine *Methode* übermittelt er seine Zugangsdaten, die ein *Provider* prüft und einem Konto mit bestimmten Rollen zuordnet.

Die Methode bestimmt den Weg der Übermittlung. Bei der Anmeldung über ein Formular legt der Server eine Sitzung an und hinterlegt sie in einem Cookie – der übliche Weg für die Nutzung im Browser. Daneben gibt es die HTTP-Basic-Authentifizierung, bei der die Zugangsdaten mit jeder Anfrage mitgeschickt werden; sie wird etwa benötigt, wenn ein Desktop-GIS auf geschützte OWS-Dienste zugreift.

Der Provider prüft die Zugangsdaten und liefert die zugehörigen Benutzerdaten und Rollen. Die Nutzer können in einer Datei hinterlegt sein oder aus einem Verzeichnisdienst wie ActiveDirectory oder LDAP stammen, dessen Gruppen auf Rollen der WebSuite abgebildet werden.

Angemeldete Nutzer behalten ihre Sitzung über einen Neustart des Servers hinweg; wie lange eine Sitzung gültig bleibt, ist konfigurierbar.

## HTTPS-Zwang der Methoden

Jede Methode gilt standardmäßig als *sicher* (`secure`) und verweigert die Anmeldung über unverschlüsseltes HTTP. Betreiben Sie die WebSuite hinter einem TLS-Proxy, ist das unkritisch; erreichen Anfragen den Server aber tatsächlich über HTTP, sind alle Methoden stillschweigend abgeschaltet und niemand kann sich anmelden. Für Tests oder ein internes Netz geben Sie mit `allowInsecureFrom` einzelne IP-Adressen frei, die auch unverschlüsselt anmelden dürfen.

## Zweiter Faktor

Verlangt ein Konto einen zweiten Faktor, steuern die Optionen des Mehr-Faktor-Adapters den Ablauf: `lifeTime` bestimmt, wie lange ein Einmalcode gültig ist (Vorgabe 120 Sekunden), `maxVerifyAttempts` die Zahl der erlaubten Fehlversuche (Vorgabe 3) und `maxRestarts`, wie oft ein Code neu angefordert werden darf (Vorgabe 0 – ein erneutes Zusenden ist also standardmäßig nicht möglich). Mit `message` hinterlegen Sie den Hinweistext, der dem Nutzer bei der Eingabe angezeigt wird.

%see
Siehe auch: [Konfiguration/Authentifizierungsmethoden](/admin-de/konfiguration/authMethod), [Konfiguration/Authentifizierungs-Provider](/admin-de/konfiguration/authProvider), [Konfiguration/Sitzungsverwaltung](/admin-de/konfiguration/authSessionManager), [Aktion/auth](/admin-de/konfiguration/action/auth).
%end
