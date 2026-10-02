# Struktur eines Modells :/admin-de/themen/daten/modelle

Ein Modell ist ein konfigurierbares Objekt, das die Attribute und das Verhalten eines Features beschreibt. Es bestimmt, welche Felder ein Feature hat, wie deren Werte ermittelt und geprüft werden und wie sie im Client bearbeitet und dargestellt werden.

Sie müssen die Felder nicht immer selbst angeben. Konfigurieren Sie keine `fields` – oder setzen Sie `withAutoFields` – liest ein Datenbank-Modell die Spalten der Tabelle aus und legt daraus automatisch passende Felder an; mit `excludeColumns` nehmen Sie einzelne Spalten aus. Ein Datenbank-Modell kann so mit minimaler Konfiguration betrieben werden.

Über `loadingStrategy` steuern Sie, wann die Features geladen werden – auf einmal, nur der sichtbare Ausschnitt (`bbox`) oder erst bei Bedarf (`lazy`).

%see
Siehe auch: [Konfiguration/Modelle](/admin-de/konfiguration/model).
%end

## Felder

Die Felder sind die Bausteine eines Modells. Jedes Feld hat einen Namen, einen Typ und einen Titel. Der Typ bestimmt die Art des Werts – etwa Text, Zahl, Datum oder Geometrie – oder eine Beziehung zu einem anderen Modell.

Ein Feld muss nicht zwingend einer Spalte entsprechen: Mit `isVirtual` legen Sie ein Feld an, das nicht in der Datenbank gespeichert, sondern nur berechnet und angezeigt wird – etwa ein zusammengesetzter Anzeigewert aus einer Wert-Definition.

%see
Siehe auch: [Konfiguration/Modell-Felder](/admin-de/konfiguration/modelField).
%end

## Werte

Ein Feld muss seinen Wert nicht unmittelbar aus der Quelle übernehmen. Über die Wert-Definition eines Feldes legen Sie fest, wie der Wert zustande kommt: als fester Vorgabewert, aus einem anderen Attribut oder aus einer Berechnung.

Wann ein Wert greift, steuern Sie getrennt für die drei Phasen `forRead`, `forCreate` und `forUpdate` (standardmäßig alle aktiv). Ob er eine Nutzereingabe ersetzt, entscheidet `isDefault`: Ein gewöhnlicher Wert **überschreibt** die Eingabe, ein Wert mit `isDefault` greift nur, **wenn der Nutzer nichts eingegeben** hat.

%see
Siehe auch: [Konfiguration/Modell-Werte](/admin-de/konfiguration/modelValue).
%end

## Validatoren

Ein Feld kann Validatoren tragen, die Eingaben prüfen, bevor ein Feature gespeichert wird – etwa auf Pflichtangaben, Wertebereiche oder ein bestimmtes Format. Schlägt eine Prüfung fehl, wird die Bearbeitung mit einer Meldung abgelehnt.

%see
Siehe auch: [Konfiguration/Modell-Validatoren](/admin-de/konfiguration/modelValidator).
%end

## Widgets

Für die Bearbeitung im Client bestimmt das Widget eines Feldes das Eingabeelement – etwa ein einfaches Textfeld, eine Auswahlliste oder einen Datumswähler. Ohne Angabe wählt der Server ein zum Feldtyp passendes Element.

%see
Siehe auch: [Konfiguration/Modell-Widgets](/admin-de/konfiguration/modelWidget).
%end

## Beziehungen

Über Beziehungsfelder verknüpft ein Modell seine Features mit denen anderer Modelle. So lassen sich zusammengehörige Objekte gemeinsam anzeigen und bearbeiten und Datenstrukturen abbilden, bei denen ein Feature auf ein oder mehrere Features eines anderen Modells verweist.

Es gibt sie in mehreren Ausprägungen – Verweis auf ein übergeordnetes Feature, Liste untergeordneter Features, Verknüpfung über eine Zwischentabelle sowie Beziehungen zu mehreren möglichen Modellen. Verknüpfte Features lassen sich beim Bearbeiten mitanlegen. Beachten Sie dabei, dass Beziehungen standardmäßig nur **eine Ebene tief** geladen und geschrieben werden: Ein verknüpftes Feature wird geladen, dessen eigene Verknüpfungen jedoch nicht automatisch mit.

%see
Siehe auch: [Konfiguration/Modell-Felder](/admin-de/konfiguration/modelField).
%end

## Tabellenansicht

Ein Modell kann seine Features nicht nur einzeln, sondern auch als Tabelle bereitstellen. In dieser Ansicht werden mehrere Features zeilenweise dargestellt und lassen sich – die entsprechenden Rechte vorausgesetzt – direkt in der Tabelle bearbeiten.

%see
Siehe auch: [Konfiguration/Modelle](/admin-de/konfiguration/model).
%end
