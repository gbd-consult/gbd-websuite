# Bauleitplanung :/user-de/menubar.bauleitplanung

Das Menü ![](bplan.svg) {title Bauleitplanung} bietet die Möglichkeit, aufbereitete Bauleit- und Landschaftspläne zu administrieren, über die GBD WebSuite darzustellen und als OGC- und  INSPIRE-konforme Dienste bereit zu stellen. Das Modul unterstützt folgende Datensätze:

| Daten                                	| Beschreibung                       						|
|---------------------------------------|-------------------------------------------------------------------------------|
| Bauleit-und Landschaftsplänen (PDF)	| Gesamtpläne im PDF-Format							|
| Planausschnitte (PNG)			| Georeferenzierte Planausschnitte mit zugehörigen World-Dateien		| 
| Planumringe (SHP)			| Polygone mit vorgegebenen Attributen Attributen				| 
| Plantabelle (Excel) (optional)	| Excel-Tabellen mit Angabe der Attribute gemäß Attributvorgaben		| 
| Planreport (txt) (optional)		| Report-Dateien zur Transformation und Entzerrung der einzelnen Pläne		| 
| Ergänzende Dokumente (PDF)		| Ergänzende Dokumente (z.B. Begründung oder Umweltbericht)			|


Die INSPIRE konforme Bereitstellung der Geodaten findet auf Anfrage dynamisch durch den GBD WebSuite Server auf Basis von XML−Schemas statt. Das bedeutet, dass keine temporären, redundanten Daten erstellt und auf dem Server abgelegt werden. Die INSPIRE konformen Dienste basieren immer auf den aktuellen Originaldaten. Für Transformation von Gauß−Krüger nach ETRS89 ist ein geeigneter Transformationsansatz integriert.

Unterstützt wird die Bereitstellung folgender INSPIRE Dienste:

- INSPIRE konformer Catalogue Service for the Web (CSW)
- INSPIRE konformer Web Mapping Service (WMS/WMTS)
- Umsetzung INSPIRE konformer Web Feature Service (WFS)
- Umsetzung INSPIRE konformer Web Coverage Service (WCS)
