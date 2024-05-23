# Bemaßung :/user-de/sidebar.bemassung

![](bemassung.png){border=1}

Das Menü ![](gbd-icon-bemassung-02.svg) {title Bemaßung} interagiert mit dem Werkzeug [Werkzeug Bemaßung](/doc/8.1/user-de/toolbar.bemassung/index.html). Es wird automatisch aktiviert, wenn Sie mit dem Werkzeug in der Karte eine erste Bemaßung gezeichnet haben und listet diese und alle weiteren im Bereich Bemaßung dieses Menüs. Dargestellt wird die Streckenlängen mit der textlichen Erweiterung. Die Funktionalität ähnelt einer technischen Bemaßung. Zum genauen Bemessen der Strecken, wird das Zeichnen durch eine Snap-Funktion unterstützt, die auf einen PostgreSQL Layer fangen kann, z.B. ALKIS und Gebäude.

Die gezeichneten Bemaßungen werden automatisch mit den jeweiligen Streckenlängen beschriftet und können frei verschoben und textlich erweitert werden. Die Eingabe von Werten ist auch mit Nachkommastellen möglich, welche dann Zentimeter darstellen.

Durch Klicken auf ![](sharp-center_focus_weak-24px.svg) {button Hinzoomen} wird auf die gewählte Bemaßung gezoomt. Durch Klicken auf ![](sharp-remove_circle_outline-24px.svg) {button Herausnehmen} wird die Bemaßung gelöscht.

![](dimensions.png){border=1}

Am unteren Menüfensterrand befindet sich weitere Funktionalität für die Ablage der ausgewählten Objekte:

| Symbol                                | Funktion                	| Funktionsbeschreibung                         |
|---------------------------------------|-------------------------------|-----------------------------------------------|
| ![](ic_folder_open_24px.svg)		| {button Auswahl laden}	| Gespeicherte Auswahlliste laden 		|
| ![](sharp-save-24px.svg)        	| {button Auswahl speichern}	| Auswahlliste speichern     			|
| ![](sharp-delete_forever-24px.svg)   	| {button Auswahl löschen}	| Aktuelle Auswahlliste löschen			|

%demo 'dimension_tool'

%info

Weitere Informationen zur [Konfiguration einer Bemaßung](/doc/8.1/admin-de/themen/bemassung/index.html) finden Sie im Administrator Handbuch.

%end
