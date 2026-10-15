# Such- und Geokodierdienste :/admin-de/themen/datenquellen/suchdienste

Manche Quellen liefern keine Karteninhalte, sondern beantworten Anfragen: Sie geben einen Ortsnamen oder eine Adresse hinein und erhalten Features mit Geometrie zurück. Solche Dienste binden Sie nicht als Layer ein, sondern als Finder für die [](/admin-de/themen/darstellung/suche); ein gleichnamiges Modell beschreibt die Attribute der Treffer. Beide sind ausschließlich lesend.

Der Dienst `nominatim` ist die Adress- und Ortssuche von OpenStreetMap. Mit `country` schränken Sie die Ergebnisse auf ein Land ein, mit `language` wählen Sie die bevorzugte Sprache der Treffer. Ohne vom Nutzer gezeichnete Geometrie ist die Suche räumlich auf das Kartenausmaß begrenzt, mit `spatialContext "view"` auf den sichtbaren Ausschnitt.

Der Dienst `gbd_geoservices` ist das Angebot von GBD Consult und beherrscht neben der Stichwortsuche auch die räumliche Suche. Er verlangt einen Zugangsschlüssel, den Sie unter `apiKey` hinterlegen.

%see
Siehe auch: [Finder/nominatim](/admin-de/konfiguration/finder/nominatim), [Modell/nominatim](/admin-de/konfiguration/model/nominatim), [Finder/gbd_geoservices](/admin-de/konfiguration/finder/gbd_geoservices), [Modell/gbd_geoservices](/admin-de/konfiguration/model/gbd_geoservices).
%end
