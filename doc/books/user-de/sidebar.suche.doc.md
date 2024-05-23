# Suche :/user-de/sidebar.suche

![](search_menu.png){border=1}

Über das Menü ![](baseline-search-24px.svg) {title Suche} können, je nach Konfiguration folgende Such-Provider (Finder) für eine Volltextsuche verwendet werden:

## Nominatim

Dieser Finder der GBD WebSuite greift auf die Such-API von OpenStreetMap zu. Sie können über eine Freiform-Suchanfrage einen Standort anhand einer Textbeschreibung oder Adresse suchen. Weitere Informationen zur Verwendung der Suche finden sich im [Nominatim Handbuch](https://nominatim.org/release-docs/develop/api/Search/#free-form-query).

Eine Abfrage über die GBD WebSuite kann unstrukturiert als "Freiformabfrage" stattfinden. Diese werden zuerst von links nach rechts und dann von rechts nach links verarbeitet, wenn dies fehlschlägt. Sie können also sowohl nach „Königsallee, Düsseldorf“ als auch nach „Düsseldorf, Königsallee“ suchen . Kommas sind optional, verbessern jedoch die Performance der Abfrage, indem sie die Komplexität der Suche verringern. Die Abfrage kann auch spezielle Ausdrücke enthalten, um den gesuchten Ort zu beschreiben oder eine Koordinate für die Suche in der Nähe einer Position.

%demo 'nominatim_search'

## Postgres

Der Postgres Finder ermöglicht das Durchsuchen von einer oder mehreren Spalten einer PostgreSQL Tabelle oder eines Views. Je nach Konfiguration wird ein exakter Suchbegriff, der Anfang, das Ende oder ein beliebiges Teilzeichen für die Suche erwartet. Das {button %-Zeichen} kann als Platzhalter genutzt werden.

%demo 'postgres_search'

%info

Weitere Informationen zur [Konfiguration der Suche](/doc/8.1/admin-de/themen/suche/index.html) finden Sie im Administrator Handbuch.

%end
