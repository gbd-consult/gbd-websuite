# Abfrage:/admin-de/plugin/abfrage

Das Abfrage-Tool der GBD WebSuite ermöglicht es Benutzern, Objektinformationen durch Klicken oder Mouseover in der Karte abzurufen. 

Die GBD WebSuite unterstützt dabei umfangreiche Möglichkeiten zur Konfiguration von Abfrage-Funktionalitäten für verschiedene Layer-Typen. Durch die Kombination von Layer-Konfiguration, Finder-Setup, Model-Definition und Template-Gestaltung können maßgeschneiderte Abfrage-Lösungen erstellt werden, die den spezifischen Anforderungen verschiedener GIS-Anwendungen gerecht werden.

Die Flexibilität des Systems ermöglicht es, sowohl einfache Punkt-und-Klick-Abfragen als auch komplexe, filterbasierte Suchen zu implementieren, wobei die Ergebnisse in benutzerdefinierten Formaten präsentiert werden können.

## Konfiguration

Diese Funktionalität kann für verschiedene Layer-Typen konfiguriert und an spezifische Anforderungen angepasst werden.

### Aktivierung der Abfrage-Funktionalität

Für jeden Layer kann die Abfrage-Funktionalität über die `withSearch` Eigenschaft aktiviert oder deaktiviert werden:

```javascript title="/data/projects/projectname.cx"
{
    type "postgres"
    title "Meine Datenbankebene"
    withSearch true  // Standardwert, aktiviert Abfragen
    // weitere Konfiguration...
}
```

### UI-Element Konfiguration

Die Abfrage-Tools werden über die Client-Konfiguration aktiviert:

```javascript title="/data/client.cx"
{
    client {
        elements+ {
            tag "Toolbar.Identify.Click"  // Objekt-Identifizierung mit Klicken
        }
        elements+ {
            tag "Toolbar.Identify.Hover"  // Objekt-Identifizierung mit Mouseover
        }
    }
}
```

## Layertyp-spezifische Konfiguration

### 1. PostgreSQL/PostGIS Layer

PostgreSQL Layer bieten umfangreiche Abfrage-Möglichkeiten mit datenbankbasierter Suche:

```javascript title="/data/projects/projectname.cx"
{
    type "postgres"
    title "Gebäude"
    dbUid "meine_datenbank"
    tableName "gebaeude"
    
    // Abfrage-Funktionalität aktivieren
    withSearch true
    
    // Finder für erweiterte Suche konfigurieren
    finders+ {
        type "postgres"
        title "Gebäude-Suche"
        tableName "gebaeude"
        dbUid "meine_datenbank"
        
        // Suchoptionen
        withKeyword true      // Stichwortsuche
        withGeometry true     // Geometrische Suche
        withFilter true       // Filtersuche
        
        // SQL-Filter für erweiterte Abfragen
        sqlFilter "status = 'aktiv'"
        
        // Modelle für Datenstruktur
        models+ {
            type "default"
            fields+ {
                name "id"
                type "integer"
                title "Gebäude-ID"
            }
            fields+ {
                name "name"
                type "text"
                title "Gebäudename"
            }
            fields+ {
                name "baujahr"
                type "integer" 
                title "Baujahr"
            }
            
            // Vorlagen für Anzeige
            templates+ {
                subject "feature.title"
                type "html"
                text "{name} (ID: {id})"
            }
            templates+ {
                subject "feature.description"
                type "html"
                text """
                <h3>{name}</h3>
                <p><strong>Baujahr:</strong> {baujahr}</p>
                <p><strong>ID:</strong> {id}</p>
                """
            }
        }
    }
}
```

### 2. GeoJSON Layer

GeoJSON Layer für dateibasierte Vektorabfragen:

```javascript title="/data/projects/projectname.cx"
{
    type "geojson"
    title "Points of Interest"
    path "/data/poi.geojson"
    
    withSearch true
    
    finders+ {
        type "geojson"
        title "POI-Suche"
        
        withKeyword true
        withGeometry true
        
        models+ {
            type "geojson"
            fields+ {
                name "name"
                type "text"
                title "Name"
            }
            fields+ {
                name "category"
                type "text"
                title "Kategorie"
            }
            
            templates+ {
                subject "feature.title"
                type "html"
                text "{name}"
            }
            templates+ {
                subject "feature.description" 
                type "html"
                text """
                <div class="poi-info">
                    <h4>{name}</h4>
                    <p><strong>Kategorie:</strong> {category}</p>
                </div>
                """
            }
        }
    }
}
```

### 3. QGIS Layer

QGIS Projekte mit Server-basierter Abfrage:

```javascript title="/data/projects/projectname.cx"
{
    type "qgis"
    title "QGIS Projekt"
    path "/data/mein_projekt.qgs"
    
    withSearch true
    
    finders+ {
        type "qgis"
        title "QGIS-Abfrage"
        
        // QGIS Provider Konfiguration
        provider {
            path "/data/mein_projekt.qgs"
            directRender ["postgres", "wfs"]
            directSearch ["postgres"]
        }
        
        // Spezifische Layer für Suche auswählen
        sourceLayers {
            names ["gebaeude_layer", "strassen_layer"]
        }
        
        withKeyword true
        withGeometry true
        withFilter true
        
        models+ {
            type "qgis"
            
            templates+ {
                subject "feature.title"
                type "html"
                text "{display_name}"
            }
            templates+ {
                subject "feature.description"
                type "html"
                text """
                <div class="qgis-feature">
                    {#each attributes}
                        <p><strong>{name}:</strong> {value}</p>
                    {/each}
                </div>
                """
            }
        }
    }
}
```

### 4. WFS Layer

Web Feature Service Layer für OGC-konforme Abfragen:

```javascript title="/data/projects/projectname.cx"
{
    type "wfs"
    title "WFS Gebäude"
    
    provider {
        url "https://example.com/wfs"
        version "2.0.0"
    }
    
    withSearch true
    
    finders+ {
        type "wfs"
        title "WFS-Gebäude-Suche"
        
        provider {
            url "https://example.com/wfs"
            version "2.0.0"
        }
        
        sourceLayers {
            names ["gebaeude"]
        }
        
        withKeyword true
        withGeometry true
        
        models+ {
            type "wfs"
            
            templates+ {
                subject "feature.title"
                type "html"
                text "{gml_id}"
            }
            templates+ {
                subject "feature.description"
                type "html"
                text """
                <table class="feature-table">
                    <tr><td>GML ID:</td><td>{gml_id}</td></tr>
                    <tr><td>Name:</td><td>{name}</td></tr>
                    <tr><td>Typ:</td><td>{building_type}</td></tr>
                </table>
                """
            }
        }
    }
}
```

### 5. WMS Layer

Web Map Service Layer mit GetFeatureInfo-Abfragen:

```javascript title="/data/projects/projectname.cx"
{
    type "wms"
    title "WMS Luftbilder"
    
    provider {
        url "https://example.com/wms"
        version "1.3.0"
    }
    
    withSearch true
    
    finders+ {
        type "wms"
        title "WMS-Abfrage"
        
        provider {
            url "https://example.com/wms"
            version "1.3.0"
        }
        
        sourceLayers {
            names ["luftbild", "orthofotos"]
        }
        
        withGeometry true
        
        models+ {
            type "wms"
            
            templates+ {
                subject "feature.description"
                type "html"
                text """
                <div class="wms-info">
                    <h4>Bildinformationen</h4>
                    <p><strong>Aufnahmedatum:</strong> {aufnahme_datum}</p>
                    <p><strong>Auflösung:</strong> {aufloesung} cm/px</p>
                </div>
                """
            }
        }
    }
}
```

### 6. Tile Layer

Für reine Tile-Layer ist normalerweise keine direkte Abfrage möglich, es sei denn, sie werden mit einem Finder kombiniert:

```javascript title="/data/projects/projectname.cx"
{
    type "tile"
    title "OpenStreetMap"
    provider {
        url "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    }
    
    // Nominatim-Finder für OSM-Daten hinzufügen
    finders+ {
        type "nominatim"
        title "OSM-Suche"
        
        withKeyword true
        withGeometry true
        
        templates+ {
            subject "feature.title"
            type "html"
            text "{display_name}"
        }
        templates+ {
            subject "feature.description"
            type "html"
            text """
            <div class="osm-result">
                <h4>{display_name}</h4>
                <p><strong>Typ:</strong> {osm_type}</p>
                <p><strong>Klasse:</strong> {class}</p>
            </div>
            """
        }
    }
}
```

## Erweiterte Optionen

### Räumlicher Kontext

Der räumliche Kontext definiert, wo Abfragen durchgeführt werden:

```javascript title="/data/projects/projectname.cx"
{
    finders+ {
        type "postgres"
        spatialContext "map"      // Standard: gesamte Karte
        // spatialContext "view"  // nur sichtbarer Bereich
        // spatialContext "selection"  // nur in Auswahl
    }
}
```

### Kategorisierung

Finder können kategorisiert werden für bessere Organisation:

```javascript title="/data/projects/projectname.cx"
{
    finders+ {
        type "postgres"
        category "Gebäude"        // Neu in Version 8.2
        title "Wohngebäude"
    }
    finders+ {
        type "postgres" 
        category "Infrastruktur"
        title "Straßen"
    }
}
```

### Berechtigungen

Abfrage-Funktionalität kann benutzerabhängig konfiguriert werden:

```javascript title="/data/projects/projectname.cx"
{
    finders+ {
        type "postgres"
        permissions {
            read "group:editors,group:viewers"
            write "group:editors"
        }
    }
}
```

## Vorlagen (Templates) für Abfrage-Ergebnisse

### Standard-Template-Typen

| Subject | Verwendung | Beschreibung |
|---------|------------|--------------|
| `feature.title` | Titel | Kurzer Titel für das Feature |
| `feature.teaser` | Vorschau | Kurzbeschreibung in Suchergebnissen |
| `feature.description` | Details | Vollständige Beschreibung im Popup |
| `feature.label` | Kartenbeschriftung | Text für Beschriftung in der Karte |

### HTML-Templates mit erweiterten Features

```javascript title="/data/projects/template/project_description.cx"
{
    templates+ {
        subject "feature.description"
        type "html"
        text """
        <div class="custom-feature-popup">
            <div class="header">
                <h3>{name}</h3>
                <span class="id">#{id}</span>
            </div>
            
            <div class="content">
                {#if description}
                    <p class="description">{description}</p>
                {/if}
                
                <table class="attributes">
                    <tr>
                        <td>Status:</td>
                        <td class="status-{status}">{status}</td>
                    </tr>
                    <tr>
                        <td>Erstellt:</td>
                        <td>{created_date|date:'dd.MM.yyyy'}</td>
                    </tr>
                </table>
                
                {#if image_url}
                    <div class="image">
                        <img src="{image_url}" alt="Bild von {name}" />
                    </div>
                {/if}
            </div>
            
            <div class="actions">
                <button onclick="zoomToFeature('{id}')">Hinzoomen</button>
                <button onclick="selectFeature('{id}')">Auswählen</button>
            </div>
        </div>
        """
    }
}
```

### CSS-Styling für Templates

```javascript title="/data/projects/projectname.cx"
{
    templates+ {
        subject "feature.description"
        type "html"
        text "..."
        
        // CSS für Template-Styling
        cssSelector ".custom-feature-popup"
    }
}
```

## Performance-Optimierung

### Limiting und Paging

```javascript title="/data/projects/projectname.cx"
{
    finders+ {
        type "postgres"
        
        // Begrenzung der Ergebnisse
        limit 100
        
        // Modell-Konfiguration für Performance
        models+ {
            type "default"
            loadingStrategy "lazy"  // Verzögertes Laden
            withTableView false     // Tabellenansicht deaktivieren falls nicht benötigt
        }
    }
}
```

### Caching

```javascript title="/data/projects/projectname.cx"
{
    type "postgres"
    withCache true    // Layer-Caching aktivieren
    
    finders+ {
        type "postgres"
        // Finder haben kein direktes Caching, aber profitieren vom Layer-Cache
    }
}
```

## Fehlerbehebung

### Häufige Probleme

1. **Keine Abfrage-Ergebnisse**
   - `withSearch true` gesetzt?
   - Finder korrekt konfiguriert?
   - Berechtigungen überprüfen

2. **Leere Popups**
   - Templates konfiguriert?
   - Feldnamen korrekt?
   - Datenquelle verfügbar?

3. **Performance-Probleme**
   - Zu viele Ergebnisse? `limit` setzen
   - Räumlichen Kontext einschränken
   - Indizes in der Datenbank prüfen

### Debug-Konfiguration

```javascript title="/data/client.cx"
{
    finders+ {
        type "postgres"
        title "Debug Finder"
        
        // Erweiterte Debugging-Optionen
        models+ {
            type "default"
            withAutoFields true    // Automatische Feldererkennung
            
            templates+ {
                subject "feature.description"
                type "html"
                text """
                <div class="debug-info">
                    <h4>Debug Information</h4>
                    <pre>{@json attributes}</pre>
                </div>
                """
            }
        }
    }
}
```

## Konfigurationsbeispiel

Hier ist ein vollständiges Beispiel einer Projekt-Konfiguration mit verschiedenen Layer-Typen und deren Abfrage-Konfiguration:

```javascript title="/data/client.cx"
{
    uid "abfrage_beispiel"
    title "Abfrage-Beispiel Projekt"
    
    map {
        crs "EPSG:3857"
        center [757072, 6663486]
        
        layers+ {
            type "tile"
            title "Basiskarte"
            provider {
                url "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            }
        }
        
        layers+ {
            type "postgres"
            title "Gebäude"
            uid "gebaeude_layer"
            dbUid "hauptdatenbank"
            tableName "gebaeude"
            
            withSearch true
            
            finders+ {
                type "postgres"
                title "Gebäude-Suche"
                category "Immobilien"
                tableName "gebaeude"
                dbUid "hauptdatenbank"
                
                withKeyword true
                withGeometry true
                withFilter true
                
                models+ {
                    type "default"
                    fields+ {
                        name "id"
                        type "integer"
                        title "ID"
                    }
                    fields+ {
                        name "name"
                        type "text"
                        title "Gebäudename"
                    }
                    fields+ {
                        name "adresse"
                        type "text" 
                        title "Adresse"
                    }
                    fields+ {
                        name "baujahr"
                        type "integer"
                        title "Baujahr"
                    }
                    
                    templates+ {
                        subject "feature.title"
                        type "html"
                        text "{name}"
                    }
                    templates+ {
                        subject "feature.description"
                        type "html"
                        text """
                        <div class="building-info">
                            <h3>{name}</h3>
                            <table>
                                <tr><td>Adresse:</td><td>{adresse}</td></tr>
                                <tr><td>Baujahr:</td><td>{baujahr}</td></tr>
                                <tr><td>ID:</td><td>{id}</td></tr>
                            </table>
                        </div>
                        """
                    }
                }
            }
        }
        
        layers+ {
            type "qgis"
            title "QGIS Projekt"
            uid "qgis_layer"
            path "/data/stadtplan.qgs"
            
            withSearch true
            
            finders+ {
                type "qgis"
                title "Stadtplan-Abfrage"
                category "Planung"
                
                provider {
                    path "/data/stadtplan.qgs"
                }
                
                sourceLayers {
                    names ["strassen", "gruenflaechen", "bebauung"]
                }
                
                withKeyword true
                withGeometry true
                
                models+ {
                    type "qgis"
                    
                    templates+ {
                        subject "feature.title"
                        type "html"
                        text "{layer_name}: {name|default:'Unbenannt'}"
                    }
                    templates+ {
                        subject "feature.description"
                        type "html"
                        text """
                        <div class="qgis-feature">
                            <h4>{layer_name}</h4>
                            {#each attributes}
                                {#if value}
                                    <p><strong>{name}:</strong> {value}</p>
                                {/if}
                            {/each}
                        </div>
                        """
                    }
                }
            }
        }
    }
    
    // Client-Konfiguration für Abfrage-Tools
    client {
        elements+ { tag "Toolbar.Identify.Click" }
        elements+ { tag "Toolbar.Identify.Hover" }
        elements+ { tag "Sidebar.Search" }
    }
}
```

