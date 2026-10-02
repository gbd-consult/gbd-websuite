# Betrieb :/admin-de/themen/betrieb

Dieses Kapitel behandelt den laufenden Betrieb: die Servermodule, die Kommandozeile und die Lokalisierung.

Es beginnt beim Host: den Containern, den beiden Verzeichnissen, die Sie beisteuern, und der Anbindung von Datenbank und QGIS-Server.

Innerhalb des Containers ist der GBD WebSuite Server kein einzelner Prozess, sondern ein Verbund mehrerer Prozesse – Webserver, Applikationsserver, Kartenproxy, Hintergrunddienste und die Konfigurationsüberwachung. Welche davon laufen und wie sie dimensioniert sind, bestimmt das Verhalten der Installation unter Last; nicht benötigte Prozesse lassen sich abschalten.

Verwaltungsaufgaben, die sich nicht konfigurieren lassen, erledigen Sie über das Kommandozeilenwerkzeug `gws` im laufenden Container – etwa das Befüllen und Löschen von Caches, das Setzen von Passwörtern oder das Indizieren von Fachdaten. Den Abschluss bildet die Lokalisierung: Sprachen, Zeitzone und die daraus abgeleiteten Formate für Datum, Uhrzeit und Zahlen.

## :/admin-de/themen/betrieb/host
## :/admin-de/themen/betrieb/server
## :/admin-de/themen/betrieb/kommandozeile
## :/admin-de/themen/betrieb/lokalisierung
