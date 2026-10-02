# Konfigurationsmodell :/admin-de/themen/grundlagen/konfiguration

Die Konfiguration ist ein Baum aus verschachtelten Objekten. Jedes Objekt hat einen Typ und eine Reihe von Eigenschaften; als Eigenschaft kommt wieder ein Objekt in Frage, eine Liste von Objekten oder ein einfacher Wert.

Viele Objekte sind Varianten eines gemeinsamen Grundtyps und wählen ihre konkrete Ausprägung über eine Typ-Eigenschaft – so entscheidet etwa bei einem Layer der Typ darüber, aus welcher Quelle er seine Daten bezieht. Welche Eigenschaften ein Objekt kennt, führt die [](/admin-de/konfiguration) vollständig auf.

Jedes Objekt trägt außerdem eine eindeutige Kennung, die *uid*, über die andere Objekte darauf verweisen – eine Datenbankverbindung etwa wird einmal definiert und anschließend von Layern, Suchen und Modellen über ihre uid genutzt. Vergeben Sie keine uid, erzeugt die WebSuite automatisch eine; Kennungen, die in URLs erscheinen, sollten Sie jedoch bewusst und dauerhaft vergeben.

Geschrieben wird die Konfiguration als JSON oder YAML. Zwei Präprozessoren erleichtern die Arbeit und erlauben es, die Konfiguration auf mehrere Dateien aufzuteilen; Syntax und Dateiaufbau beschreiben die [](/admin-de/erste-schritte/konfigurationsgrundlagen).
