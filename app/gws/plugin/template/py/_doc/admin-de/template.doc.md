# Vorlage "py" :/admin-de/konfiguration/template/py

Die Vorlage `py` ist ein Python-Modul, das die Ausgabe programmatisch erzeugt. Das Modul muss eine Funktion `main` bereitstellen, die die Argumente entgegennimmt und ein Antwort-Objekt zurückgibt. Den Pfad zur Moduldatei geben Sie über `path` an.

## Beispiel-Konfiguration ::

```javascript
templates+ {
    subject "feature.title"
    type "py"
    path "/data/templates/feature_title.py"
}
```

Das `subject` legt den Zweck der Vorlage fest, hier den Titel eines Features. Über `path` verweisen Sie auf ein Python-Modul, das eine Funktion `main` bereitstellt; diese erhält die Argumente des Aufrufs und gibt ein Antwort-Objekt zurück.

%ref "gws.plugin.template.py.Config"
