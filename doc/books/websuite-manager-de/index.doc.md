# GBD WebSuite Manager Benutzerhandbuch :/websuite-manager 

Benutzerhandbuch für das GBD WebSuite Manager Plugin

**Übersicht**


Das ![](gws_logo-24px.svg) GBD WebSuite Manager Plugin ist eine Erweiterung die in QGIS installiert werden kann. Es ermöglicht die einfache Bereitstellung von QGIS Projekten in der GBD WebSuite. Voraussetzung ist, dass ein Nutzeraccount zu einer GBD WebSuite vorhanden ist, und dass deren GBD WebSuite Manager Schnittstelle aktiv ist.

![](uebersicht_f.png)

**Installation**


Es gibt zwei Möglichkeiten, das Plugin in QGIS zu installieren. Zum einen können Sie es von unserem `Plugin Repository <https://plugins.gbd-consult.de/>`_ herunterladen und als ZIP-Datei in QGIS einbinden.

Zum anderen ist eine direkte Einbindung unseres Plugin-Repositorys in QGIS über folgenden Link möglich:

![](repodetails.png)

Wenn das Plugin installiert ist, ist es in QGIS unter Web -> GBD WebSuite -> GBD WebSuite Manager zu finden.
Alternativ kann es in den Werkzeugkästen ausgewählt und somit prominent in die Werkzeugleiste integriert werden.
Die einzelnen Funktionen werden im jeweiligen Menüpunkt genauer erklärt.

**Anmelden**

Um das GBD WebSuite Plugin nutzen zu können muss man sich auf einem GBD WebSuite Server, mit aktivierter GBD WebSuite Manager Schnittstelle, authentifizieren.
Für diese Authentifizierung wird das QGIS eigene Authentifizierungssystem genutz.

Zuerst, falls noch nicht vorhanden, muss unter ``Einstellungen -> Optionen -> Authentifizierung`` ein Hauptpasswort gesetzt werden. Danach kann man über ``Neue Authentifizierungskonfiguration hinzufügen`` eine neue Verbindungen erstellen. Für eine Verbindung muss ein Name, Nutzername, Passwort und Serveradresse gesetzt werden, wobei Nutzername, Passwort und Serveradresse den Daten der GBD WebSuite Installation entsprechen müssen.

![](anmeldung.png)

Die Authentifizierung sollte nun unter dem Drop Down Menü auswählbar sein und, bei Auswahl, automatisch eine Verbindung zur GBD WebSuite herstellen.
Sollten mehrere GBD WebSuite Installationen vorhanden sein, können auch mehrere Authentifizierungen hinterlegt werden, und zwischen diesen gewechselt werden.

![](authentifizierung.png)

Das Plugin prüft automatisch die Authentifizierung und stellt, bei Erfolg, das Plugin auf aktiv und läd die vorhandenen Projekte.

%info
Weitere Informationen bezüglich des QGIS Authentifizierungssystems erhalten Sie in der QGIS Dokumentation: https://docs.qgis.org/3.16/de/docs/user_manual/auth_system/index.html.
%end

**Aktuelles Projekt**


Hier können Sie ihr aktuelles QGIS Projekt in die GBD WebSuite hochladen.
Es werden sämtliche Vektorlayer und Dienste mit implementiert.

![](selected_project_no_options.png)

Tragen Sie den gewünschten Titel ein, der in die GBD WebSuite übernommen werden soll.
Wenn alles angepasst ist, muss man nur noch auf ![](mActionAdd.svg) ``Aktuelles Projekt hinzufügen`` klicken und das Projekt wird direkt in die GBD WebSuite hochgeladen.

**Vorhandene Projekte verwalten**

![](project_list.png)

Anhand dieser Liste kann man eine Übersicht über die hochgeladenen Projekte gewinnen.
Das gewählte Projekt kann man über ![](gws_logo-24px.svg) ``Öffne die WebSuite`` sich in der WebSuite anzeigen lassen.
Über ![](link_24px.svg) ``Link anzeigen`` kann man sich den Projektlink oder die URLs der OGC-Dienste anzeigen lassen,
über die das Projekt in andere Anwendungen integriert werden können.

Drei Werkzeuge am unteren Rand.
Über ![](mActionHelpContents.png) ``Hilfe`` gelangt man zur ausführlichen Hilfe und Dokumentation.
Durch Klicken des Button ![](mActionFileOpen.png) ``ausgewähltes Projekt anzeigen`` öffnet man das gewählte Projekt lokal in QGIS.
Mit Hilfe des ![](mActionTrash.png) Button kann das gewählte Projekt vom Server gelöscht werden.
