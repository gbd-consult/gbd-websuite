# Such- und Geokodierdienste :/admin-de/themen/datenquellen/suchdienste

Manche Quellen liefern keine Karteninhalte, sondern beantworten Anfragen: Sie geben einen Ortsnamen oder eine Adresse hinein und erhalten Features mit Geometrie zurück. Solche Dienste binden Sie nicht als Layer ein, sondern als Finder für die [](/admin-de/themen/darstellung/suche); ein gleichnamiges Modell beschreibt die Attribute der Treffer. Beide sind ausschließlich lesend.

Der Dienst `nominatim` ist die Adress- und Ortssuche von OpenStreetMap. Mit `country` und `language` schränken Sie die Ergebnisse auf ein Land und eine Sprache ein – ohne diese Angaben durchsucht der Dienst die ganze Welt und liefert entsprechend unspezifische Treffer.

Der Dienst `gbd_geoservices` ist das Angebot von GBD Consult und beherrscht neben der Stichwortsuche auch die räumliche Suche. Er verlangt einen Zugangsschlüssel, den Sie unter `apiKey` hinterlegen.

%see
Siehe auch: [Finder/nominatim](/admin-de/konfiguration/finder/nominatim), [Modell/nominatim](/admin-de/konfiguration/model/nominatim), [Finder/gbd_geoservices](/admin-de/konfiguration/finder/gbd_geoservices), [Modell/gbd_geoservices](/admin-de/konfiguration/model/gbd_geoservices).
%end
