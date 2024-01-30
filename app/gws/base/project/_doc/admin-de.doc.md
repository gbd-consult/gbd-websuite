# Projekte :/admin-de/config/projekte

%reference_de 'gws.base.project.core.Config'

Ein *Projekt* (``project``) in der GBD WebSuite besteht aus einer Karte (``map``), optionale Druck-Konfiguration (``print``) und zusätzlichen Optionen. In Abschnitten ``api`` und ``client`` können Sie die im App-Konfig definierte Aktionen und Client-Optionen überschreiben bzw. erweitern. Mittels ``access`` können Sie die Zugriffsrechte zu Projekten steuern.

Mit der Aktion ``project`` werden Projekte für den GWS Client freigeschaltet. Wenn diese Aktion fehlt, können Projekte nicht im Client aufgerufen werden, können aber für andere Zwecke wie z.B. ein WMS Dienst verwender werden.

## Projekt-Vorlagen

Im Projekt kann eine Info-Vorlage konfiguriert werden, die im Client gezeigt werden sobald der Nutzer die Projekt-Eigenschaften aktiviert. Diese Vorlage muss ``subject`` ``project.description`` haben (siehe [Vorlagen](/admin-de/config/template)).
