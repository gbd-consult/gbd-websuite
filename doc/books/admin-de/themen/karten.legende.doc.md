# Legende :/admin-de/themen/karten/legende

Für einen Layer kann eine Legende festgelegt werden. Ist keine definiert, verwendet die WebSuite die für den Layer-Typ passende Standardlegende – bei einem WMS etwa die vom Dienst gelieferte Grafik. Eine Legende kann ebenso abgeschaltet, durch ein festes Bild ersetzt oder über eine Vorlage frei gestaltet werden, wenn eine einfache Grafik nicht ausreicht.

Eine einmal erzeugte Legendengrafik hält der Server bis zum nächsten Neustart oder Neuladen vor. Legendenbilder von einem entfernten Dienst oder vom QGIS-Server werden zusätzlich im Cache-Verzeichnis gespeichert, für die Dauer von `cacheMaxAge` (Vorgabe ein Tag); ändert sich eine solche Legende, wird die neue Fassung erst sichtbar, wenn diese Zeit abgelaufen ist und der Server neu gestartet wurde.

%see
Siehe auch: [Konfiguration/Legenden](/admin-de/konfiguration/legend).
%end
