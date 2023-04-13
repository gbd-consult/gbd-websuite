# html :/admin-de/config/template/type/

HTML-Vorlagen können die Befehle der Vorlagen-Sprache enthalten, beliebige HTML Formatierung und spezielle HTML Tags die unter ^print beschrieben sind.

Zum Beispiel, hier ist eine Vorlage fur die Layer-Beschreibung (`subject: layer.description`):

    <h1>{layer.title}</h1>

    <p>{layer.meta.abstract}</p>

    @if layer.has_legend
        <img src="_?cmd=mapHttpGetLegend&layerUid={layer.uid}"/>
    @end

    <ul>
        @each layer.meta.keywords as keyword
            <li>{keyword}</li>
        @end
    </ul>

Beschreibung (`subject: feature.description`) eines "city" Feature, welches die Attribute "name", "area" und "population" besitzt:

    @if population > 100000
        <div class="big-city">{name}</div>
    @else
        <div class="small-city">{name}</div>
    @end

    <p> <strong>Area:</strong> {area} </p>
    <p> <strong>Population:</strong> {population} </p>

^NOTE Das erste Zeichen (ausgenommen Whitespace) der Ausgabe einer HTML-Vorlage muss `<` sein, ansonsten wird die Vorlage als `text` interpretiert.
