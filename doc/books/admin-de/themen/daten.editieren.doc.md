# Editieren :/admin-de/themen/daten/editieren

Mit der Editierfunktion können Nutzer Vektorobjekte im Client zeichnen, ändern und mit Attributen versehen; die Änderungen werden in die Datenbank zurückgeschrieben.

Ob ein Datenmodell bearbeitet werden darf, legt es selbst fest: Ein Datenbank-Modell wird über die Eigenschaft `isEditable` zum Editieren freigegeben. Welche Nutzer welche Operationen ausführen dürfen, steuern die Berechtigungen des Modells getrennt nach Anlegen, Ändern und Löschen (`permissions.create`, `permissions.write` und `permissions.delete`).

Innerhalb eines Modells bestimmen die Felder das Bearbeiten im Detail. Ein Feld kann editierbar oder nur lesbar sein, kann Eingaben über Validatoren prüfen und über ein Widget ein passendes Eingabeelement vorgeben. Über Beziehungen zwischen Modellen sind auch anspruchsvollere Abläufe möglich – etwa das Auswählen eines verknüpften Features aus einem anderen Modell, während ein Feature bearbeitet wird.

%see
Siehe auch: [Aktion/edit](/admin-de/konfiguration/action/edit), [Konfiguration/Modelle](/admin-de/konfiguration/model).
%end
