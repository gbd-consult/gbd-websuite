# Editieren :/user-de/sidebar.editieren

![](editing.png){border=1}

Über das Menü ![](sharp-edit-24px.svg) {title Editieren} können unterschiedliche Datentypen erstellt, bearbeitet und gelöscht werden. Dies umfasst das Abbilden komplexer Datenstrukturen und Beziehungen zwischen Objekten. Die Eingabe von Informationen kann durch eine {button flexibele Validierung}, Standardwerte und Platzhalter unterstützt werden und findet über Formulare statt. Ergänzt werden können Objekte in der Karte durch {button Bilder und Dokumente}, die in einer {button Dateiverwaltung} gemanaged werden. 

**Datentypen**

| Datentyp					| Beschreibung                         										|
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [Geometriedaten](geometriedaten)		| Punkte-, Linien- und Polygon-Objekte mit oder ohne Attributen können erstellt, bearbeitet und gelöscht werden	|
| [Tabellen](tabellen)       			| Objekte in Datentabellen ohne Geometrien (Attributtabellen) können erstellt, bearbeitet und gelöscht werden	|
| [Dokumente und Bilder](dokumente-und-bilder)	| Bilder und Dokumente können hochgeladen, verknüpft und gelöscht werden					|

**Komplexe Datenstrukturen und Beziehungen**

In der GBD WebSuite können miteinander in Beziehung stehende Daten modelliert werden. Für diese Aufgabe wird {button SQLAlchemy} verwendet. Diese Bibliothek ermöglicht es, komplexe Datenmodelle zu erstellen und Datenbankoperationen effizient durchzuführen. Ein Beispiel für ein Datenmodell mit 1:M, M:1 und M:N Beziehungen findet sich in der folgenden Abbildung.

![](komplexe_beziehungen.png){border=1, width=1000px}

**Flexible Validierung**

Beim Editieren werden Daten in Formularfelder eingetragen. Um Fehler beim Eintragen zu vermeiden, können die Formularfelder konfiguriert werden. Es kann eine Datenvalidierung definiert werden, in der geprüft wird, ob eine Kombination von Werten einer Reihe zulässiger Kombinationen angehört. Desweiteren können Vorgabewerte und Platzhalter vorgegeben werden.

![](validierung_de.png){border=1, width=1000px}

**Bilder und Dokumente**

Jedem Objekt einer Karte können Bilder und Dokumente hinzugefügt werden. Diese werden in einer Dateiverwaltung gemanaged und als Vorschau beim Editieren im Formular angezeigt.

![](bilder_dokumente_de.png){border=1, width=1000px}

## Geometriedaten

Geometriedaten mit dazugehörigen Attributen können in der GBD WebSuite erstellt, bearbeitet und gelöscht werden. Das Modul {title Editieren} der Seitenleiste bietet einen Ablauf, den wir im Weiteren beschreiben. 

**Datenlayer auswählen**

Mit einem Klick auf das Menü ![](sharp-edit-24px.svg) {title Editieren} öffnet sich eine Liste von Layern, die über die GBD WebSuite bearbeiten werden können. Im Bild sind der Polygon-Layer {button Districts} und der Punkt-Layer {button Points of interest} zu sehen. 

![](digitalisieren1.png){border=1, width=1000px}

Wenn man das Icon ![](database_24px.svg) {title Bearbeiten} links neben einem Layernamen anklickt, offnet sich eine neue Ebene, in der alle Objekte des Layers aufgelistet sind. In der nächsten Abbildung sind es die Stadtteile (Districts) der Stadt Düsseldorf.

![](digitalisieren2.png){border=1, width=1000px}

Wenn man das Icon ![](table_24px.svg) {title Attribittabelle öffnen} rechts neben einem Layernamen anklickt, offnet sich ein Dialog, der die Attribute aller Objekte des Layers auflistet sind. Je nach Konfiguration können die Felder bearbeitet werden können.

![](digitalisieren4.png){border=1, width=1000px}

Oberhalb der Objektliste befindet sich eine Suchleiste. Darüber können Sie nach Objekten in der Liste suchen. Mit einem Klicke auf den Button ![](sharp-center_focus_weak-24px.svg) {button Hinzoomen} kann auf das jeweilige Objekt in der Karte gezoomt werden.

Am unteren Menüfensterrand befindet sich weitere Funktionalität:table_24px.svg

| Symbol				| Funktion             				| Funktionsbeschreibung	                					|
|---------------------------------------|-----------------------------------------------|-------------------------------------------------------------------------------|
| ![](database_24px.svg)		| {button Zur Editieren Übersicht}		| Objekt in der Karte auszuwählen und dessen Geometrie und Attribute bearbeiten	|
| ![](table_24px.svg)		   	| {button Attribittabelle öffnen}		| Attributtabelle eines Layers oder eines Layerobjektes öffnen			|
| ![](draw-black-24px.svg) 		| {button Neues Objekt zeichnen}		| Neues Objekt (Punkt, Linie oder Polygon) in der Karte zeichnen	    	|
| ![](sharp-control_point-24px.svg)	| {button Neues Objekt über Formular erstellen}	| Neues Objekt mit/ohne Geometrie über ein Formular erstellen   	 	|

Mit dem Symbol ![](draw-black-24px.svg) {button Neues Objekt zeichnen} kann ein Objekt im Kartenfenster ausgewählt werden. Dieses wird dann markiert mit seinen Stützpunkten dargestellt und die dazugehörigen Objekteigenschaften werden im Menüfenster angezeigt. Nun können die Geometrien und Attribute editiert werden. Folgende Möglichkeiten stehen zur Verfügung:

**Vorhandene Geometrien editieren**

* Durch Anklicken eines Objekts in der Karte mit gedrückter, linker Maustaste, kann das Objekt (Punkt, Linie, Polygon) verschoben werden.
* Nähert man sich mit dem Mauszeiger dem Stützpunkt eines Objekts, wird der Mauszeiger gefangen. Mit gedrückter, linker Maustaste, kann der Stützpunkt verschoben werden
* Nähert man sich mit dem Mauszeiger dem Stützpunkt eines Objekts, wird der Mauszeiger gefangen. Ein Klick mit der linken Maustaste löscht den Stützpunkt
* Nähert man sich mit dem Mauszeiger dem Segment eines Objekts, wird der Mauszeiger gefangen. Ein Klick mit der linken Maustaste fügt einen neuen Stützpunkt hinzu.

![](digitalisieren3.png){border=1}

**Attribute editieren**

* Bei Punkten kann die Position, alternativ zum Verschieben in der Karte, auch über die Felder "X" und "Y" als Koordinatenwert angepasst werden. 

**Neue Geometrien erstellen**

Mit dem Symbol ![](draw-black-24px.svg) {button Zeichnen} kann ein neues Geometrieobjekt (Punkt, Linie, Polygon) erstellt werden. 

* {title Punkt}: Mit der linken Maustaste wird ein Klick in die Karte gesetzt. Danach öffnen sich die Objekteigenschaften im Menüfenster, um Attribute zu vergeben.
* {title Linie}: Mit der linken Maustaste wird ein erster Stützpunkt in die Karte gesetzt. Mit jedem weiteren Klick der linken Maustaste wird ein weiterer Stützpunkt der Linie in die Karte gesetzt. Mit einem Doppelklick wird das Zeichnen des Linienobjekts abgeschlossen. Danach öffnen sich die Objekteigenschaften im Menüfenster, um Attribute zu vergeben.
* {title Polygon}: Mit der linken Maustaste wird ein erster Stützpunkt in die Karte gesetzt. Mit jedem weiteren Klick der linken Maustaste wird ein weiterer Stützpunkt des Polygons in die Karte gesetzt. Mit einem Doppelklick wird das Zeichnen des Polygons abgeschlossen und die Fläche geschlossen. Danach öffnen sich die Objekteigenschaften im Menüfenster, um Attribute zu vergeben.

Mit dem Symbol ![](sharp-control_point-24px.svg) {button Hinzufügen} können neue Punktgeometrien durch die Eingabe von X- und Y-Koordinaten erstellt werden. 

Mit dem Symbol ![](baseline-close-24px.svg) {button Beenden} navigiert man zurück zur Layeransicht.

**Neue Geometrien über ein Formular erstellen**

![](geometrie_mit_texteditor.png){border=1}


## Tabellen


## Dokumente und Bilder


