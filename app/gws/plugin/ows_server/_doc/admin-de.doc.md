# OWS Dienste :/admin-de/plugin/ows

Die GBD WebSuite kann als OWS (OGC Web Services) Server fungieren. Sie können diese Dienste für jedes Projekt frei konfigurieren. Wenn mehrere Versionen für einen OWS unterstützt werden, antwortet der Server immer mit der Version die angefragt wird, d.h. wenn z.B. http://host.de/wms?service=WMS&version=1.1.1 aufgerufen wird, erfolgt die Ausgabe in der Version 1.1.1.

## Aktion ``ows``

%reference_de 'gws.plugin.ows_service.action.Config'

Die Dienste werden freigeschaltet indem Sie die Aktion ``ows`` global oder in einem Projekt konfigurieren. Diese Aktion besitzt eine Liste von Diensten (``service``), mit der Sie die konkreten Dienste konfigurieren.

Für alle OWS Dienste muss aus den Projekt-Layern ein "root" Layer ausgewählt sein. Sie können diesen Layer explizit mit der ``root`` Eigenschaft konfigurieren. Alternativ wird der erste Layer (der auf der obersten Ebene gelegene) gewählt.

## Unterstützte Dienste

Derzeit sind folgende Dienste implementiert:

### WMS

%reference_de 'gws.ext.ows.service.wms.Config'

Der WMS-Dienst ist vollständig gemäß der Eigenschaften der Version ``1.1.0``, ``1.1.1`` und ``1.3.0`` implementiert.

### WFS

%reference_de 'gws.plugin.ows_service.wfs.Config'

Der WFS-Dienst ist gemäß der Eigenschaften der Version ``2.0`` implementiert. Derzeit unterstützen wir nur folgenden Funktionen:

- ``GetCapabilities``
- ``DescribeFeatureType``
- ``GetFeature`` mit der ``Envelope`` Operation

%info
 In der Zukunft, planen wir das "Basic WFS" Profil sowie WFS 3.0 umzusetzen.
%end

### WCS

%reference_de 'gws.plugin.ows_service.wcs.Config'

Es werden WCS Versionen ``1.0.0`` und ``2.0.1`` unterstützt.

### WMTS

%reference_de 'gws.plugin.ows_service.wmts.Config'

Es wird WMTS Version ``1.0.0`` unterstützt.

### CSW Dienst

%reference_de 'gws.plugin.ows_service.csw.Config'

Die GBD WebSuite enthält eine Basis-Implementation eines CSW Dienstes. Dieser Dienst kann nur in der App-Konfig konfiguriert werden. Derzeit sind folgende Operationen implementiert:

- ``GetCapabilities``
- ``DescribeRecord``
- ``GetRecords``, mit ``PropertyIsLike`` und ``Envelope`` Filtern
- ``GetRecordById``

Sie können auch zwischen Metadata-Profilen ``iso`` (ISO 19139) oder ``dcmi`` (Dublin Core) wählen.

Der CSW Dienst ist für alle OWS und ISO Metadaten zuständig. Sobald Sie den Dienst aktivieren, werden alle im System vorhandene Metadaten gesammelt und als CSW Einträge (``record``) dargestellt. Jedes Objekt bekommt automatisch eine ``MetadataURL``, die auf die entsprechende CSW Seite zeigt, sofern Sie unter ``meta.url`` nichts anderes angeben.

## Layer-Konfiguration

%reference_de 'gws.plugin.ows_service.core.LayerConfig'

Sie können bei jedem Layer mit ``enabled`` definieren, ob dieser für OWS Dienste berücksichtigt werden soll. Mit ``enabledServices`` legen Sie eine engere Auswahl an OWS Diensten fest, in denen der Layer gezeigt werden soll. Sie können weiter den Namen der Layer ``name`` und die Features, die sich auf dem Layer befinden ``featureName`` definieren.

Externe WMS/WFS Layer werden automatisch "kaskadiert".

## URL rewriting

Standardmäßig werden OWS-Dienste unter einer dynamischen URL angezeigt, die die Dienst ``uid`` sowie Projekt ``uid`` enthält, z.B.:

    http://example.com/_?cmd=owsHttpService&uid=my_wms_service&projectUid=meinprojekt

Sie können diese URL in eine schönere Form mit URL-Rewriting umschreiben, z.B.:

    https://example.com/my_wms_service/meinprojekt

Damit Ihre in Capabilities Dokumenten angegebene URLs auch "schön" aussehen, müssen Sie auch das ``reversedRewrite`` konfigurieren. Siehe [Web-Server](/admin-de/config/web) für Details.

## Vorlagen

Die für einen Dienst notwendige XML Dokumente werden vom System automatisch erstellt. Sie haben jedoch die Möglichkeit, diese Dokumente anzupassen. Dafür definieren Sie unter Dienst ``templates`` eine Vorlage mit dem ``subject`` das für eine der folgenden Dokument-Kategorien steht:

| Subject | Dienste |
|---|---|
| ``ows.GetCapabilities`` | alle |
| ``ows.GetFeatureInfo`` | WMS, WFS |
| ``ows.DescribeFeatureType`` | WFS |
| ``ows.DescribeCoverage`` | WCS |
| ``ows.DescribeRecord`` | CSW |
| ``ows.GetRecords`` | CSW |
| ``ows.GetRecordById`` | CSW |

Die Vorlagen müssen in ``xml`` bzw ``text`` Format sein, siehe [Vorlagen](/admin-de/config/template) für mehr Infos.

## INSPIRE Support

Die GBD WebSuite unterstützt die Europäische Direktive INSPIRE indem die ``GetCapabilities`` Dokumente INSPIRE Metadaten enthalten. Es ist auch möglich INSPIRE-Konforme (*harmonisierte*) Sachdaten in ``GetFeature`` Responses auszugeben.

### INSPIRE Metadaten

Um einen INSPIRE Meta-Block (``inspire_vs:ExtendedCapabilities``) in Ihren ``GetCapabilities`` Dokument einzubauen, setzen Sie die Eigenschaft ``withInspireMeta`` auf ``true`` und befüllen Sie die notwendigen Eigenschaften in Dienst Metadaten.

### INSPIRE Harmonisierung

Derzeit können Sie INSPIRE Sachdaten erstellen indem Sie eine dedizierte Vorlage für ``ows.GetFeatureInfo`` einbauen die anhand von Quell-Feature Attributen eine INSPIRE-konforme Struktur generiert.
