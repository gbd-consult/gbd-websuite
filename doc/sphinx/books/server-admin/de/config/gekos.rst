GekoS Integration
=================

^REF gws.ext.action.gekos.Config

Die GBD WebSuite kann mit dem System "GekoS Bau+" der Firma GekoS (https://www.gekos.de) integriert werden. Dafür muss eine Aktion vom Typ ``gekos`` im Abschnitt ``actions`` eingetragen werden.

Gek-Online Anbindung
--------------------

Die GBD WebSuite kann die Daten aus dem Modul Gekos-Online in eine Postgis Tabelle übertragen. Der Export erfolgt mit dem Kommandozeilen Befehl ``gws gekos load``.

Es gibt folgende Konfigurationsoptionen um den Export anzupassen

{TABLE}
``crs`` | KBS für GekoS Daten
``instances`` | Liste der vorhandenen Gek-Online Instanzen
``params`` | Parameter für Gek-Online XML Aufrufe
``position`` | Position-Korrektur für Punkte
``table`` | Postgis Tabelle
``url`` | Gek-Online Web Adresse
{/TABLE}

GekoS Konfiguration
-------------------

Für die visuelle Anbindung an GekoS Programm muss in der GBD WebSuite ein Projekt angelegt werden (im folgenden "gekos_project"). Im GekoS Programm, unter "GIS Integration" müssen dann folgende Werte eingetragen werden:

{TABLE}
``GIS-URL-Base`` | Ihre GWS-Server Adresse, z.B. ``http://example.com``
``GIS-URL-ShowXY`` | ``/project/gekos_project?x=<x>&y=<y>&z=1000``
``GIS-URL-ShowFs`` | ``/project/gekos_project?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>``
``GIS-URL-GetXYFromMap`` | ``/project/gekos_project?&x=<x>&y=<y>&gekosUrl=<returl>``
``GIS-URL-GetXYFromFs`` | ``/_/?cmd=gekosHttpGetXy&alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>``
``GIS-URL-GetXYFromGrd`` | ``/_/?cmd=gekosHttpGetXy&alkisAd=<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>``
{/TABLE}

(Es wird angenommen, dass Sie die "project" Rewrite-Regel nutzen, s. dazu ^client).

Zusätzlich muss im Client das Element ``Toolbar.Gekos`` aktiviert werden.
