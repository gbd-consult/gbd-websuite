# Berechtigungen :/admin-de/themen/zugriff/berechtigungen

Nach der Anmeldung entscheidet die *Autorisierung*, ob ein Nutzer ein Objekt in gewünschter Weise nutzen darf. Grundlage sind Rollen und Zugriffsregeln.

Jeder Nutzer trägt eine Menge von Rollen – teils vom System vergeben, teils von dem Provider, der ihn bei der [](/admin-de/themen/zugriff/authentifizierung) geprüft hat. Einige Rollen sind fest vorgegeben: Gäste, angemeldete Nutzer, die Gesamtheit aller Nutzer für öffentliche Objekte sowie eine Administratorrolle mit uneingeschränktem Zugriff.

Für die meisten Objekte – Projekte, Layer, Modelle, Aktionen und weitere – lassen sich Berechtigungen festlegen, getrennt nach Lesen und den schreibenden Operationen. Eine Berechtigung ist eine Folge von Regeln, die einer Rolle den Zugriff erlauben oder verwehren; sie werden der Reihe nach geprüft, bis eine Rolle des Nutzers zutrifft. Fehlt an einem Objekt eine Regel, gilt die des übergeordneten Objekts – und am Wurzelobjekt wird der Zugriff andernfalls verweigert. So vererben sich Rechte entlang des Konfigurationsbaums: Sie legen sie an zentraler Stelle fest und schränken einzelne Objekte gezielt ein.

Je nachdem, ob eine Installation überwiegend öffentlich oder überwiegend geschützt ist, beginnen Sie mit einer freigebenden oder einer sperrenden Grundregel und weichen für einzelne Objekte davon ab. Beachten Sie dabei, dass die für die Anmeldung nötigen Funktionen für alle zugänglich bleiben müssen.
