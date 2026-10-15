# Helfer "edit" :/admin-de/konfiguration/helper/edit

Der Hilfsdienst `edit` stellt die serverseitige Editier-Unterstützung bereit. Er liest, schreibt und löscht Features editierbarer Modelle und liefert der Aktion `edit` die passenden Modelle und verknüpfbaren Objekte. Der Dienst besitzt keine eigene Konfiguration und wird automatisch angelegt, sobald eine Aktion ihn benötigt, etwa die Aktion `edit`; die Editier-Funktionen richten Sie über die Aktion `edit` und die als `isEditable` markierten Modelle ein.
