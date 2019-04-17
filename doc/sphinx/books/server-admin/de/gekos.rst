GekoS Integration
=================

Das Werkzeug "GekoS Bau+" von der Firma GekoS (https://www.gekos.de) kann in die GBD WebSuite integriert werden. Dafür muss eine Aktion vom Typ ``gekos`` im Abschnitt ``actions`` eingetragen werden.


GekoS Konfiguration
-------------------

Für die visuelle Anbindung an das GekoS Programm muss in der GBD WebSuite ein Projekt angelegt werden (im folgenden "gekos_project"). Im GekoS Programm, unter "GIS Integration" müssen dann folgende Werte eingetragen werden:

TABLE

GIS-URL-Base  ~ Ihre GBD WebSuite-Server Adresse
GIS-URL-ShowXY  ~ ``/project/gekos_project?x=<x>&y=<y>&z=1000``
GIS-URL-ShowFs ~ ``/project/gekos_project?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>``
GIS-URL-GetXYFromMap ~ ``/project/gekos_project?&x=<x>&y=<y>&gekosUrl=<returl>``
GIS-URL-GetXYFromFs   ~ ``/_/?cmd=gekosHttpGetXy&alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>``
GIS-URL-GetXYFromGrd  ~ ``/_/?cmd=gekosHttpGetXy&alkisAd=<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>``

/TABLE


Gek-Online Anbindung
--------------------

Die GBD WebSuite kann die Daten aus dem Modul Gek-Onlne in eine PostGIS Tabelle übertragen. Der Export erfolgt mit dem Kommandozeilen Befehl ::

    gws gekos load


Es gibt folgende Konfigurationsoptionen um den Export anzupassen:

TABLE
url ~ Gek-Online Web Adresse
params ~ Parameter für Gek-Online XML Aufrufe
crs ~ KBS für GekoS Daten
instances ~ Liste der vorhandenen Gek-Online Instanzen
table ~ Postgis Tabelle
position ~ Position-Korrektur für Punkte
/TABLE
