# Helfer "upload" :/admin-de/konfiguration/helper/upload

Der Hilfsdienst `upload` verwaltet stückweise (chunked) Datei-Uploads. Der Client sendet eine Datei in einzelnen Teilstücken; der Dienst setzt sie zwischengespeichert wieder zusammen und stellt die fertige Datei zur weiteren Verarbeitung bereit. Er wird von Aktionen genutzt, die Datei-Uploads entgegennehmen, besitzt keine eigene Konfiguration und wird bei Bedarf automatisch bereitgestellt.
