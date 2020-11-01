OWS Dienste
===========

Die GBD WebSuite kann als OWS (OGC Web Services) Server fungieren. Sie können diese Dienste für jedes Projekt frei konfigurieren.

Aktion ``ows``
--------------

^REF gws.ext.action.ows.Config

Die Dienste werden freigeschaltet indem Sie die Aktion ``ows`` global oder in einem Projekt konfigurieren. Diese Aktion besitzt eine Liste von Diensten (``service``), wo Sie die konkrete Dienste konfigurieren.

Für alle OWS Dienste muss aus Projekt-Layer ein "root" Layer ausgewählt sein. Sie können diesen Layer explizit mit der ``root`` Eigenschaft konfigurieren, per Default wird den ersten Layer auf der obersten Ebene genommen.

Unterstützte Dienste
--------------------

Derzeit werden folgende Dienste unterstützt:

wms
~~~

^REF gws.ext.ows.service.wms.Config

Der WMS-Dienst sind vollständig gemäß der Eigenschaften der Version 1.3.0 implementiert.

wfs
~~~

^REF gws.ext.ows.service.wfs.Config

Der WFS-Dienst ist gemäß der Eigenschaften der Version 2.0 implementiert. Derzeit unterstützen wir nur folgenden Funktionen:

- ``GetCapabilities``
- ``DescribeFeatureType``
- ``GetFeature`` mit der ``Envelope`` Operation

^NOTE In der Zukunft, planen wir das "Basic WFS" Profil sowie WFS 3.0 umzusetzen.

wcs
~~~

^REF gws.ext.ows.service.wcs.Config

Es werden WCS Versionen ``1.0.0`` und ``2.0.1`` unterstützt.

wmts
~~~~

^REF gws.ext.ows.service.wmts.Config

Es wird WMTS Version ``1.0.0`` unterstützt.

CSW Dienst
----------

^REF gws.ext.ows.service.csw.Config

GWS enthält eine Basis-Implementation von einem CSW Dienst. Dieser Dienst kann nur in der App-Konfig konfiguriert werden. Derzeit sind folgende Operationen implementiert:

- ``GetCapabilities``
- ``DescribeRecord``
- ``GetRecords``, mit ``PropertyIsLike`` und ``Envelope`` Filtern
- ``GetRecordById``

Sie können auch zwischen Metadata-Profilen ``iso`` (ISO 19139) oder ``dcmi`` (Dublin Core) wählen.

CSW Dienst ist für alle OWS und ISO Metadaten zuständig. Sobald Sie den Dienst aktivieren, werden alle im System vorhandene Metadaten gesammelt und als CSW Einträge (``record``) dargestellt. Jedes Objekt bekommt automatisch eine ``MetadataURL``, die auf die entsprechende CSW Seite zeigt.

Layer-Konfiguration
-------------------

^REF gws.common.layer.types.OwsConfig

Zusätzlich zur der Aktion Konfiguration, können Sie bei jedem Layer definieren, ob dieser Layer überhaupt für OWS Dienste berücksichtigt wird (``enabled``), wenn ja, für welche (``enabledServices``) und welchen Namen der Layer selbst (``name``) und die Features, die sich auf diesem Layer befinden (``featureName``) haben.

Externe WMS/WFS Layer werden automatisch "kaskadiert".

URL rewriting
-------------

Standardmäßig werden OWS-Dienste unter einer dynamischen URL angezeigt, die die Dienst ``uid`` sowie Projekt ``uid`` enthält, z.B. ::

    http://example.com/_?cmd=owsHttpService&uid=my_wms_service&projectUid=meinprojekt

Wenn Sie diese URL in eine schönere Form mit URL-Rewriting umschreiben, z.B: ::

    https://example.com/my_wms_service/meinprojekt

Damit Ihre in Capabilities Dokumenten angegebene URLs auch "schön" aussehen, müssen Sie auch reversierte Rewriting (``reversedRewrite``) konfigurieren. Siehe ^web für Details.

Vorlagen
--------

Die für einen Dienst notwendige XML Dokumente werden vom System automatisch erstellt. Sie haben jedoch die Möglichkeit, diese Dokumente anzupassen. Dafür definieren Sie unter Dienst ``templates`` eine Vorlage mit dem ``subject`` das fîr eine der folgenden Dokument-Kategorien steht:

{TABLE head}
Subject | Dienste
``ows.GetCapabilities`` | alle
``ows.GetFeatureInfo`` | WMS, WFS
``ows.DescribeFeatureType`` | WFS
``ows.DescribeCoverage`` | WCS
``ows.DescribeRecord`` | CSW
``ows.GetRecords`` | CSW
``ows.GetRecordById`` | CSW
{/TABLE}

Die Vorlagen müssen in ``xml`` bzw ``text`` Format sein, s. ^template für mehr Info.

INSPIRE Support
---------------

GWS unterstützt Europäische Direktive INSPIRE indem die ``GetCapabilities`` Dokumente INSPIRE Metadaten enthalten. Es ist auch möglich INSPIRE-Konforme (*harmonisierte*) Sachdaten in ``GetFeature`` Responses auszugeben.

INSPIRE Metadaten
~~~~~~~~~~~~~~~~~

Um einen INSPIRE Meta-Block (``inspire_vs:ExtendedCapabilities``) in Ihren ``GetCapabilities`` Dokument einzubauen, setzen Sie die Eigenschaft ``withInspireMeta`` auf ``true`` und befüllen Sie die notwendigen Eigenschaften in Dienst Metadaten.

INSPIRE Harmonisierung
~~~~~~~~~~~~~~~~~~~~~~

Derzeit können Sie INSPIRE Sachdaten erstellen indem Sie eine dedizierte Vorlage für ``ows.GetFeatureInfo`` einbauen die anhand von Quell-Feature Attributen eine INSPIRE-konforme Struktur generiert.
