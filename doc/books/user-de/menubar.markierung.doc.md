# Markierungen :/user-de/menubar.marking

Das Menü ![](gbd-icon-markieren-messen-01.svg) {title Markierungen} interagiert mit dem Werkzeug ![](gbd-icon-markieren-messen-01.svg) {button Markieren und Messen}.

Die Objekte, die man mit Hilfe des Werkzeuges erzeugt hat, werden hier aufgelistet. Über ![](sharp-center_focus_weak-24px.svg) {button } kann man zu dem gezeichneten Objekt hineinzoomen, über |delete| können Sie das gezeichnete Objekt direkt löschen.

![](marking.png)


Wenn ein Objekt neu gezeichnet oder im Menü ![](gbd-icon-markieren-messen-01.svg) {title Markierungen} ausgewählt wurde, öffnet sich das jeweilige Objektfenster. Hier können Geometrieeigenschaften abgelesen und Beschriftung eingetragen werden. Unter dem Reiter {title Platzhalter} befinden sich vorgefertigte Ausdrücke welche häufig zum Beschreiben von Geometrien verwendet werden. Wenn keine Beschriftung eingetragen wird, werden Punkte, Linien und Flächen standardmäßig mit Koordinaten, Streckenlänge in Kilometer oder Fläche in Quadratkilometer beschriftet. Falls dies auch nicht gewünscht ist, muss lediglich die Formel in dem Beschriftungsfenster gelöscht werden.

![](measure_info.png)

**Platzhalter für die Beschriftung**

| Feld               | Erläuterung               	|
|--------------------|----------------------------------|
| {x}                | gibt die X-Koordinate an        	|
| {y}                | gibt die Y-Koordinate an        	|
| {width}            | gibt die Breite an              	|
| {height}           | gibt die Höhe                   	|
| {len}              | gibt die Länge einer Linie an   	|
| {area}             | gibt die Fläche an              	|
| {radius}           | gibt den Radius an              	|


Für die Verwendung der Platzhalter wurde eine Formel entwickelt, wodurch die Einheit und die Präzision individuell bestimmt werden können. Diese Formel besitzt folgende Form { |  | }. In den ersten Abschnitt wird der gewünschte Platzhalter eingetragen. In den zweiten Abschnitt wird die Einheit eingetragen. Der letzte Abschnitt bestimmt die Präzision bzw wie viele Nachkommastellen vergeben werden sollen.

**Beispiele**

| {x | km | 2}   | x Position in Kilometer, 2 Dezimalstellen|
|----------------|------------------------------------------|
| {x | dms }     | x Position in Dezimalstellen             |
| {area | ha | 3}| Fläche in ha, 3 Dezimalstellen           |
| {area | km}    | Fläche in km2, 0 Dezimalstellen          |


**Darstellungskonfigurationen**

Über den Punkt ![](brush.svg) {Darstellung} gelangt man zu den ausführlichen Darstellungsoptionen für ![](gws_digits1_24px.svg) Geometrie und ![](gws_digits2_24px.svg) Beschriftung. Beide Darstellungen können über den jeweils obersten Menüpunkt an- oder ausgeschaltet werden. Bei den Geometrieoptionen kann der Stil der Linien, sowie die Füllung der Geometrie eingestellt werden. Die Darstellungsoptionen für die Beschriftung bieten die gleichen Möglichkeiten.
Zusätzlich kann jedoch über die Menüpunkte ``Platzierung``, ``Ausrichtung`` und ``Versatz X/Y`` die Beschriftung positioniert werden.

![](measure_combi.png)

Mit Hilfe des Werkzeuges ![](cursor.svg) {Bearbeiten} können erstellte Markierungen in der Karte angewählt und danach bearbeitet werden. Eine neue Markierung kann über das ![](sharp-gesture-24px.svg) Icon angelegt werden oder über das Werkzeug ![](gbd-icon-markieren-messen-01.svg) {button Markieren und Messen}. Das Laden ![](gbd-icon-ablage-oeffnen-01.svg) und Abspeichern ![](sharp-save-24px.svg) der gewählten Markierungen ist ebenfalls möglich. Über das Werkzeug ![](sharp-delete_forever-24px.svg) {Alle löschen} werden alle Markierung auf einmal gelöscht.

