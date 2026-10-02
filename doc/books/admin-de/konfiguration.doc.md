# Konfiguration :/admin-de/konfiguration

Dieser Teil beschreibt die Konfigurationsobjekte der GBD WebSuite, geordnet nach Kategorien. Eine Kategorie fasst Objekte zusammen, die dieselbe Aufgabe erfüllen – etwa alle Layer-Typen oder alle Authentifizierungs-Provider.

Die meisten Kategorien umfassen mehrere *Typen*. Welchen Typ Sie verwenden, geben Sie über die Eigenschaft `type` an; sie entscheidet, welche weiteren Eigenschaften das Objekt kennt. Jede Typ-Seite beschreibt den Zweck des Typs und seine wichtigsten Eigenschaften.

Einige Abschnitte – etwa Client, CSS-Stile, Host-Konfiguration und Web – kennen keine Typen. Sie beschreiben je einen zusammenhängenden Bereich der Konfiguration, der sich nicht über eine `type`-Eigenschaft verzweigt.

Für die Zusammenhänge statt einzelner Optionen führen die [](/admin-de/themen) in jeden Bereich ein; die vollständige, automatisch erzeugte Liste aller Objekte und Eigenschaften bietet die [](/admin-de/reference).

## Aktionen :action

Eine Aktion ist eine Funktionsgruppe des Servers, die im Rahmen der [](/admin-de/themen/grundlagen/architektur) dynamische Anfragen beantwortet – etwa das Rendern der Karte, eine Suche oder einen Druckauftrag. Aktionen stehen nur zur Verfügung, wenn sie in der Applikation oder im Projekt aufgeführt sind; fehlt eine Aktion, ist die zugehörige Funktion abgeschaltet.

### :/admin-de/konfiguration/action/*

## :/admin-de/konfiguration/application

## Authentifizierungs-Provider :authProvider

Ein Authentifizierungs-Provider prüft bei der [](/admin-de/themen/zugriff/authentifizierung) die Zugangsdaten und liefert das zugehörige Benutzerkonto mit seinen Rollen. Sie konfigurieren einen oder mehrere Provider global in der Applikation, je nachdem, wo Ihre Benutzer verwaltet werden – in einer Datei, einer Datenbank oder einem Verzeichnisdienst.

### :/admin-de/konfiguration/authProvider/*

## Authentifizierungsmethoden :authMethod

Eine Authentifizierungsmethode bestimmt, auf welchem Weg ein Nutzer bei der [](/admin-de/themen/zugriff/authentifizierung) seine Zugangsdaten übermittelt: über ein Anmeldeformular mit Sitzungs-Cookie, über HTTP-Basic bei jeder Anfrage oder über ein Token. Welche Methoden Sie aktivieren, hängt davon ab, ob nur der Browser-Client zugreift oder auch externe Programme wie ein Desktop-GIS.

### :/admin-de/konfiguration/authMethod/*

## :/admin-de/konfiguration/client

## :/admin-de/konfiguration/style

## Datenbank-Provider :databaseProvider

Ein Datenbank-Provider beschreibt eine Verbindung zu einer [Datenbank](/admin-de/themen/daten/datenbanken); unterstützt wird PostgreSQL/PostGIS. Sie definieren die Verbindung einmal global und sprechen sie anschließend über ihre uid aus Layern, Modellen und Findern an. Die Zugangsdaten hinterlegen Sie am besten außerhalb der Konfiguration in einer `pg_service.conf`.

### :/admin-de/konfiguration/databaseProvider/*

## :/admin-de/konfiguration/printer

## Exporter :exporter

Ein Exporter wandelt Features beim [](/admin-de/themen/publishing/export) in ein Ausgabeformat um, das Nutzer herunterladen können – etwa CSV, GeoJSON, GML, KML oder Shapefile. Global konfigurierte Exporter stehen allen Projekten zur Verfügung, im Projekt konfigurierte nur diesem.

### :/admin-de/konfiguration/exporter/*

## Finder :finder

Ein Finder durchsucht bei der [](/admin-de/themen/darstellung/suche) eine bestimmte Quelle und liefert die Treffer als Features zurück. Sie konfigurieren Finder global, je Projekt oder an einem Layer; der Server fragt alle zuständigen Finder ab und führt deren Ergebnisse zu einer Liste zusammen.

### :/admin-de/konfiguration/finder/*

## Helfer :helper

Ein Helfer ist ein globales Hilfsobjekt, das eine Funktion für mehrere andere Objekte bereitstellt – etwa den E-Mail-Versand, das Hochladen von Dateien oder die [](/admin-de/themen/darstellung/datenablage) des Clients. Helfer werden nicht unmittelbar aufgerufen, sondern von Aktionen und Modellen im Hintergrund genutzt; Sie konfigurieren sie einmal in der Applikation.

### :/admin-de/konfiguration/helper/*

## :/admin-de/konfiguration/host

## :/admin-de/konfiguration/map

## Kommandozeilen-Befehle :cli

Die Kommandozeilen-Befehle beschreiben, was Sie mit dem Werkzeug `gws` im laufenden Container tun können: den Cache verwalten, Passwörter setzen, den Server steuern oder Fachdaten indizieren. Anders als die übrigen Kategorien werden diese Objekte nicht konfiguriert, sondern über die [](/admin-de/themen/betrieb/kommandozeile) aufgerufen.

### :/admin-de/konfiguration/cli/*

## Layer :layer

Ein [](/admin-de/themen/karten/layer) ist eine Ebene der Karte; sein Typ bestimmt, woher das Kartenmaterial stammt – aus einem QGIS-Projekt, einer Datenbank, einem OGC-Dienst, einem Kacheldienst oder einer Datei. Layer werden in der Karte eines Projekts konfiguriert und lassen sich über Gruppen zu einem Baum ordnen. Rasterlayer liefern ein fertiges Pixelbild, Vektorlayer einzelne Objekte mit Geometrie und Attributen.

### :/admin-de/konfiguration/layer/*

## Legenden :legend

Eine [](/admin-de/themen/karten/legende) erklärt die Darstellung eines Layers. Sie konfigurieren sie an einem Layer, wenn die Standardlegende seines Typs nicht genügt – etwa um ein festes Bild zu hinterlegen, mehrere Legenden zusammenzufassen oder die Legende über eine Vorlage frei zu gestalten.

### :/admin-de/konfiguration/legend/*

## Modell-Felder :modelField

Ein Modell-Feld beschreibt ein Attribut eines Features: seinen Namen, seinen Titel und seinen Typ. Der Typ bestimmt die Art des Werts – Text, Zahl, Datum, Geometrie oder Datei – oder eine Beziehung zu einem anderen Modell. Zusammen mit Werten, Validatoren und Widgets bilden die Felder ein [Modell](/admin-de/themen/daten/modelle).

### :/admin-de/konfiguration/modelField/*

## Modell-Validatoren :modelValidator

Ein Validator prüft die Eingabe eines Feldes, bevor ein Feature gespeichert wird – auf Pflichtangaben, Wertebereiche oder ein bestimmtes Format. Sie konfigurieren Validatoren an den Feldern eines [Modells](/admin-de/themen/daten/modelle); schlägt eine Prüfung fehl, wird die Bearbeitung mit einer Meldung abgelehnt.

### :/admin-de/konfiguration/modelValidator/*

## Modell-Werte :modelValue

Eine Wert-Definition legt fest, wie der Wert eines Feldes zustande kommt, wenn er nicht unmittelbar aus der Quelle stammt: als fester Vorgabewert, aus dem angemeldeten Nutzer, aus einer Formatvorlage oder aus einer Berechnung. Sie konfigurieren sie an einem Feld eines [Modells](/admin-de/themen/daten/modelle), getrennt danach, ob der Wert beim Lesen oder beim Schreiben eines Features gilt.

### :/admin-de/konfiguration/modelValue/*

## Modell-Widgets :modelWidget

Ein Widget bestimmt, mit welchem Eingabeelement ein Feld eines [Modells](/admin-de/themen/daten/modelle) im Client bearbeitet wird – Textfeld, Auswahlliste, Datumswähler, Datei-Upload oder die Auswahl eines verknüpften Features. Sie konfigurieren es am Feld; ohne Angabe wählt der Server ein zum Feldtyp passendes Element.

### :/admin-de/konfiguration/modelWidget/*

## Modelle :model

Ein [Modell](/admin-de/themen/daten/modelle) beschreibt, welche Attribute die [Features](/admin-de/themen/daten/features) einer Quelle besitzen und wie diese gelesen, geprüft und geschrieben werden. Sie konfigurieren Modelle an Vektorlayern, an Findern oder global; der Typ richtet sich nach der Datenquelle.

### :/admin-de/konfiguration/model/*

## Multifaktor-Adapter :authMultiFactorAdapter

Ein Multifaktor-Adapter fügt der [](/admin-de/themen/zugriff/authentifizierung) einen zweiten Faktor hinzu – einen Einmalcode per E-Mail oder aus einer Authenticator-App. Sie konfigurieren die Adapter global; ob ein Konto einen zweiten Faktor verwendet, ergibt sich aus den Angaben des Kontos.

### :/admin-de/konfiguration/authMultiFactorAdapter/*

## OWS-Dienste :owsService

Ein OWS-Dienst stellt die Daten eines Projekts über eine standardisierte [OGC-Schnittstelle](/admin-de/themen/publishing/ows) bereit, sodass externe Anwendungen sie nutzen können: WMS und WMTS für Kartenbilder, WFS für Vektorobjekte, CSW für Metadaten. Dienste lassen sich global oder je Projekt konfigurieren; welche Layer darin erscheinen, steuern Sie am Layer und über die Berechtigungen.

Alle Dienste erzeugen ihre XML-Dokumente selbst, lassen sich darin aber über Vorlagen anpassen. Welche Vorlage wofür verwendet wird, bestimmt ihr *Subject* nach dem Muster `ows.<Operation>` – etwa `ows.GetCapabilities`. Welche Subjects ein Dienst unterstützt, steht auf seiner Seite.

### :/admin-de/konfiguration/owsService/*

## :/admin-de/konfiguration/project

## Sitzungsverwaltung :authSessionManager

Die Sitzungsverwaltung speichert die Sitzungen angemeldeter Nutzer, sodass eine [Anmeldung](/admin-de/themen/zugriff/authentifizierung) einen Neustart des Servers übersteht. Sie konfigurieren sie global und legen dort insbesondere fest, wie lange eine Sitzung gültig bleibt.

### :/admin-de/konfiguration/authSessionManager/*

## Speicher-Provider :storageProvider

Ein Speicher-Provider bestimmt, wo die im Client erzeugten Objekte – Markierungen, Bemaßungen, Auswahllisten – in der serverseitigen [](/admin-de/themen/darstellung/datenablage) liegen. Voreingestellt ist eine lokale SQLite-Datei; einen anderen Provider konfigurieren Sie, wenn die Daten in einer gemeinsamen Datenbank liegen sollen.

### :/admin-de/konfiguration/storageProvider/*

## Vorlagen :template

Eine Vorlage erzeugt eine dynamische Ausgabe – eine HTML-Seite, den Text eines Feature-Pop-ups, eine Legende oder ein Druckdokument. Über das *Subject* einer [Vorlage](/admin-de/themen/darstellung/templates) bestimmen Sie, wofür die WebSuite sie einsetzt; für jeden Zweck gibt es eine Standardvorlage, die Sie ersetzen können.

### :/admin-de/konfiguration/template/*

## :/admin-de/konfiguration/web
