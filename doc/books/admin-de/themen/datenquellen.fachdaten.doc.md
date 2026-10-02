# Fachdatenbestände :/admin-de/themen/datenquellen/fachdaten

Einige Datenbestände haben eine feste fachliche Struktur, die die WebSuite kennt. Sie binden diese nicht über einen Layer-Typ ein, sondern über eine Aktion, die die Struktur auswertet und im Client eine passende Oberfläche anbietet. Die Datengrundlage müssen Sie dafür vorher aufbauen; das Kapitel [](/admin-de/themen/fachmodule) beschreibt die Module im Einzelnen.

Der wichtigste Fall ist ALKIS. Die WebSuite durchsucht Daten des Amtlichen Liegenschaftskatasterinformationssystems und bietet eine Flurstückssuche an. Voraussetzung ist eine PostgreSQL/PostGIS-Datenbank, die aus den ALKIS-Quelldaten im NAS-Format aufgebaut wurde – die Rohdaten selbst liest die WebSuite nicht.

Daneben gibt es Anbindungen an Drittsysteme, bei denen Daten nicht bezogen, sondern ausgetauscht werden. Das Bauverwaltungssystem [](/admin-de/themen/fachmodule/gekos) verknüpft seine Vorgänge wechselseitig mit der Karte, und über [](/admin-de/themen/fachmodule/qfieldcloud) gelangen im Feld erfasste Daten zurück in die Datenbank.

%see
Siehe auch: [Aktion/alkis](/admin-de/konfiguration/action/alkis), [Aktion/gekos](/admin-de/konfiguration/action/gekos), [Aktion/qfieldcloud](/admin-de/konfiguration/action/qfieldcloud).
%end
