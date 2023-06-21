# Markieren & Messen :/user-de/toolbar.markieren-messen

Das Werkzeug ![](gbd-icon-markieren-messen-01.svg) {title Markieren & Messen} kann mit Hilfe von fünf verschiedenen Geometrischen Formen angewendet werden.

![](markingddasdasd.png)

Die ![](g_point.svg) {button Punktmarkierung} dient dem Markierungszweck. Ein Messen ist hier nicht möglich.
Aktivieren Sie das Werkzeug und klicken Sie mit der linken Maustaste auf den gewünschten Punkt in der Karte.
Sofort wird ein Punkt gesetzt, welcher standardmäßig mit X- und Y-Koordinate beschriftet wird. Sie können die Beschriftung beliebig anpassen.
Nutzen Sie dazu das sich automatisch öffnende {title Markierung bearbeiten}-Menü in der Seitenleiste.
Ausführlichere Darstellungskonfigurationen für Geometrie und Beschriftung sind unter ![](brush.svg) { button Stil } vorhanden.

Bei der ![](g_line.svg) {button Distanzmessung} wird mit der linken Maustaste auf die Karte ein Startpunkt und mit jedem weiteren Klick ein weiterer Punkt gesetzt.
Mit einem Doppelklick wird der Endpunkt gesetzt und die Länge der Strecke angezeigt.

Für die ![](g_box.svg) {button Rechtecksflächenmessung} müssen zwei Punkte gesetzt werden. Erzeugen Sie den ersten Punkt mit der linken Maustaste.
Ziehen Sie nun das Rechteck über die gewünschte Fläche in der Karte. Durch ein weiteres Klicken mit der linken Maustaste wird das Rechteck festgestellt.
Nun öffnet sich automatisch wieder das Fenster in der Seitenleiste.
Hier kann jetzt neben der X- und Y-Koordinate auch die Breite und Höhe des gezeichneten Rechtecks abgelesen werden.
Diese Werte dienen zur standardmäßigen Beschriftung. Über das Beschriftungsfeld kann der Text nach Belieben angepasst werden.

Bei der ![](g_poly.svg) {button Polygonflächenmessung} kann durch das Setzen mehrerer Punkte ein Polygon gezeichnet werden.
Starten Sie das Werkzeug und setzen Sie mit der linken Maustaste einen Startpunkt. Jetzt können beliebig viele Punkte hinzugefügt werden.
Durch einen Doppelklick wird das Zeichnen beendet und das Polygon abgeschlossen. Die Beschriftung kann erneut über das dazugehörige Menüfenster vergeben werden.
Falls nichts Individuelles gewählt wird, wird das Polygon mit der Flächengröße beschriftet.

Bei einer ![](g_circle.svg) {button Kreisflächenmessung} wird zuerst ein Punkt mit einem Klick in die Karte gesetzt, dies ist der Kreismittelpunkt.
Mit einem zweiten Klick um den Kreis, wird der Radius festgelegt. Es wird nun ein Kreis gezeichnet, welcher mit dem Radius in Metern beschriftet wird.
Der Radius sowie die Beschriftung können nachträglich verändert werden. Hierzu nutzen Sie das entsprechende Feld in dem sich geöffneten Menüfenster.

Über die Symbole in der Werkzeugleiste können Sie zusätzlich das Zeichnen jeglischer Geometrie bestätigen TODO! {button Speichern } oder ![](baseline-close-24px.svg) {button Abbrechen }.
Mit ![](zoom-in-24px.svg) {button Hinzoomen} kann auf jedes Objekt gezoomt werden.

**Eigenschaften**

Jedes Objekt kann nachträglich angepasst werden. Wählen Sie dazu das Objekt im Menü ![](gbd-icon-markieren-messen-01.svg) {title Markierungen} an und es öffnen sich automatisch die Objekteigenschaften.
Nun können Sie die vorhanden Stützpunkte wieder bewegen und durch einen Doppelklick neue Stützpunkte setzen.
Bei der Kreisflächenmessung kann der Radius angepasst werden.
Bei jedem Objekt kann über den Reiter ``Platzhalter`` frei gewählt werden, ob die Längen in Meter oder Kilometer angegeben werden.
Dabei ist die Eingabe von Werten mit Nachkommastellen möglich, welche je nach gewählter Einheit dann Zentimeter (cm) oder Meter(m) darstellen.

![](measure_edit.png)

**Platzhalter für die Beschriftung**

| **Feld** |**Erläuterung** |
|---|---|
| {x} | gibt die X-Koordinate an |
| {y} | gibt die Y-Koordinate an |
| {widt} | gibt die Breite an |
| {height} | gibt die Höhe |
| {len} | gibt die Länge einer Linie an |
| {area} | gibt die Fläche an |
| {radius} | gibt den Radius an |


Für die Verwendung der Platzhalter wurde eine Formel entwickelt, wodurch die Einheit und die Präzision individuell bestimmt werden können.
Diese Formel besitzt folgende Form { |  | }. In den ersten Abschnitt wird der gewünschte Platzhalter eingetragen. In den zweiten Abschnitt wird die Einheit eingetragen.
Der letzte Abschnitt bestimmt die Präzision bzw wie viele Nachkommastellen vergeben werden sollen.

Hier ein paar Beispiele:

| Formel | Bedeutung |
|---|---|
| \{x \| km \| 2 \} | x Position in Kilometer, 2 Dezimalstellen |
| \{x \| dms \} | x Position in Dezimalstellen |
| \{area \| ha \| 3 \} | Fläche in ha, 3 Dezimalstellen |
| \{area \| km \} | Fläche in km2, 0 Dezimalstellen |


**Darstellungskonfigurationen**

Über den Punkt |style| ``Darstellung`` gelangt man zu den ausführlichen Darstellungsoptionen für ![](gws_digits1_24px.svg) Geometrie und ![](gws_digits2_24px.svg) Beschriftung.
Beide Darstellungen können über den jeweils obersten Menüpunkt an- oder ausgeschaltet werden.
Bei den Geometrieoptionen kann der Stil der Linien, sowie die Füllung der Geometrie eingestellt werden.
Die Darstellungsoptionen für die Beschriftung bieten die gleichen Möglichkeiten.
Zusätzlich kann jedoch über die Menüpunkte ``Platzierung``, ``Ausrichtung`` und ``Versatz X/Y`` die Beschriftung positioniert werden.

![](measure_geo_text.png)

%info
 Wie andere Werkzeuge auch, kann dieses Werkzeug individuell angepasst werden.
 Die Platzierung kann verändert werden, sowie es ebenfalls möglich wäre, zum Beispiel nur die Flächenmessung zu aktivieren.
%end