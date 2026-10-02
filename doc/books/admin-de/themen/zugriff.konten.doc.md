# Benutzerkonten :/admin-de/themen/zugriff/konten

Über die Authentifizierung hinaus kann die GBD WebSuite Benutzerkonten selbst verwalten – anlegen, bearbeiten, sperren und per E-Mail freischalten. Grundlage ist der Hilfsdienst `account`; darauf setzen eine administrative Oberfläche, ein Selbstbedienungs-Dialog und ein passender Authentifizierungs-Provider auf.

## Kontentabelle

Die Konten liegen in einer Datenbanktabelle, deren Name frei wählbar ist. Sie muss eine feste Menge von Spalten enthalten, über die die WebSuite Status, Passwort und die Freischaltung verwaltet: `email`, `status`, `password`, `mfauid`, `mfasecret` sowie `tc`, `tctime` und `tccategory` (temporärer Code für die Freischaltung). Weitere Spalten für Stammdaten sind erlaubt und werden über die Datenmodelle des Hilfsdienstes bearbeitbar gemacht.

Jedes Konto durchläuft drei Zustände: *neu* (angelegt, aber noch nicht freigeschaltet), *onboarding* (Freischaltung läuft) und *aktiv*.

## Konten anlegen und freischalten

Es gibt **keine** offene Selbstregistrierung. Konten werden von Berechtigten über die administrative Oberfläche angelegt (Aktion `accountadmin`, Client-Element `Sidebar.AccountAdmin`) und zunächst im Zustand *neu* gespeichert.

Freigeschaltet wird ein Konto über ein *Onboarding*: Die WebSuite erzeugt einen temporären Code, verschickt ihn per E-Mail an die hinterlegte Adresse (die Adresse legen Sie mit `onboardingUrl` fest) und der Nutzer setzt darüber sein Passwort und richtet – falls konfiguriert – einen zweiten Faktor ein. Danach ist das Konto *aktiv*. Diesen Selbstbedienungs-Teil stellt die Aktion `account` bereit. Der Code ist nur begrenzt gültig (`tcLifeTime`, Vorgabe eine Stunde).

Auch ein Passwort-Wechsel läuft über dieses Onboarding: Ein Zurücksetzen – durch einen Administrator oder über den Kommandozeilen-Befehl `gws account reset` – erzeugt einen neuen Code und verschickt erneut eine Freischaltungs-Mail. Eine anonyme „Passwort vergessen"-Selbstbedienung gibt es nicht.

Der E-Mail-Versand setzt einen konfigurierten E-Mail-Helfer voraus. Melden sich die freigeschalteten Nutzer später an, prüft der Authentifizierungs-Provider `account` ihre Zugangsdaten gegen dieselbe Tabelle.

%see
Siehe auch: [Aktion/account](/admin-de/konfiguration/action/account), [Aktion/accountadmin](/admin-de/konfiguration/action/accountadmin), [Helfer/account](/admin-de/konfiguration/helper/account), [Authentifizierungs-Provider/account](/admin-de/konfiguration/authProvider/account).
%end
