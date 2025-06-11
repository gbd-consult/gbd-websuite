"""Various inspire-related data."""


# Metadata

from enum import Enum
from typing import Optional


class IM_DegreeOfConformity(Enum):
    """Degree to which the dataset complies with INSPIRE implementing rules."""

    conformant = 'conformant'
    notConformant = 'notConformant'
    notEvaluated = 'notEvaluated'


class IM_ResourceType(Enum):
    """Type of resource as defined by INSPIRE."""

    dataset = 'dataset'
    series = 'series'
    service = 'service'


class IM_SpatialDataServiceType(Enum):
    """Classification of spatial data services according to INSPIRE."""

    discovery = 'discovery'
    view = 'view'
    download = 'download'
    transformation = 'transformation'
    invoke = 'invoke'
    other = 'other'


class IM_SpatialScope(Enum):
    """Spatial scope of the resource according to INSPIRE."""

    national = 'national'
    regional = 'regional'
    local = 'local'
    european = 'european'
    global_ = 'global'


class IM_Theme(Enum):
    """INSPIRE data themes."""

    addresses = 'addresses'
    administrativeUnits = 'administrativeUnits'
    agriculturalAndAquacultureFacilities = 'agriculturalAndAquacultureFacilities'
    areaManagementRestrictionRegulationZonesAndReportingUnits = 'areaManagementRestrictionRegulationZonesAndReportingUnits'
    atmosphericConditions = 'atmosphericConditions'
    bioGeographicalRegions = 'bioGeographicalRegions'
    buildings = 'buildings'
    cadastralParcels = 'cadastralParcels'
    coordinateReferenceSystems = 'coordinateReferenceSystems'
    elevation = 'elevation'
    energyResources = 'energyResources'
    environmentalMonitoringFacilities = 'environmentalMonitoringFacilities'
    geographicalGridSystems = 'geographicalGridSystems'
    geographicalNames = 'geographicalNames'
    geology = 'geology'
    habitatsAndBiotopes = 'habitatsAndBiotopes'
    humanHealthAndSafety = 'humanHealthAndSafety'
    hydrography = 'hydrography'
    landCover = 'landCover'
    landUse = 'landUse'
    meteorologicalGeographicalFeatures = 'meteorologicalGeographicalFeatures'
    mineralResources = 'mineralResources'
    naturalRiskZones = 'naturalRiskZones'
    oceanographicGeographicalFeatures = 'oceanographicGeographicalFeatures'
    orthoimagery = 'orthoimagery'
    populationDistributionDemography = 'populationDistributionDemography'
    productionAndIndustrialFacilities = 'productionAndIndustrialFacilities'
    protectedSites = 'protectedSites'
    seaRegions = 'seaRegions'
    soil = 'soil'
    speciesDistribution = 'speciesDistribution'
    statisticalUnits = 'statisticalUnits'
    transportNetworks = 'transportNetworks'
    utilityAndGovernmentServices = 'utilityAndGovernmentServices'


class IM_MandatoryKeyword(Enum):
    """Mandatory keywords for INSPIRE services."""

    infoMapAccessService = 'infoMapAccessService'
    infoFeatureAccessService = 'infoFeatureAccessService'
    infoCoverageAccessService = 'infoCoverageAccessService'
    infoSensorAccessService = 'infoSensorAccessService'
    infoProductAccessService = 'infoProductAccessService'
    infoFeatureTypeService = 'infoFeatureTypeService'
    infoPropertyTypeService = 'infoPropertyTypeService'
    infoCatalogueService = 'infoCatalogueService'
    infoRegistryService = 'infoRegistryService'
    infoGazetteerService = 'infoGazetteerService'
    infoOrderHandlingService = 'infoOrderHandlingService'
    infoStandingOrderService = 'infoStandingOrderService'


TAGS = {
    'au:AdministrativeBoundary': {},
    'au:AdministrativeUnit': {},
    'us-govserv:GovernmentalService': {
        'name': 'str',
        'address': 'str',
        'email': 'str',
        'fax': 'str',
        'phone': 'str',
        'serviceType': ('select', 'http://inspire.ec.europa.eu/codelist/ServiceTypeValue'),
    },
    'plu:SpatialPlan': {
        # @TODO
    },
}

# THEMES are generated with
#
# import requests, json
#
# langs = 'en', 'de'
# es = {}
#
# for lang in langs:
#     js = requests.get(f'http://inspire.ec.europa.eu/theme/theme.{lang}.json').json()
#     for p in js['register']['containeditems']:
#         p = p['theme']
#         id = p['id'].split('/')[-1]
#         es.setdefault(id, {})
#         es[id][lang] = [p['label']['text'], p['definition']['text']]
#
# print(json.dumps(dict(sorted(es.items())), indent=4, ensure_ascii=False))


THEMES = {
    'ac': {
        'en': [
            'Atmospheric conditions',
            'Physical conditions in the atmosphere. Includes spatial data based on measurements, on models or on a combination thereof and includes measurement locations.',
        ],
        'de': [
            'Atmosphärische Bedingungen',
            'Physikalische Bedingungen in der Atmosphäre. Dazu zählen Geodaten auf der Grundlage von Messungen, Modellen oder einer Kombination aus beiden sowie Angabe der Messstandorte.',
        ],
    },
    'ac-mf': {
        'en': [
            'Atmospheric Conditions and meteorological geographical features',
            'Physical conditions in the atmosphere. Includes spatial data based on measurements, on models or on a combination thereof and includes measurement locations. Weather conditions and their measurements; precipitation, temperature, evapotranspiration, wind speed and direction.',
        ],
        'de': [
            'Atmospheric Conditions and meteorological geographical features',
            'Physical conditions in the atmosphere. Includes spatial data based on measurements, on models or on a combination thereof and includes measurement locations. Weather conditions and their measurements; precipitation, temperature, evapotranspiration, wind speed and direction.',
        ],
    },
    'ad': {
        'en': ['Addresses', 'Location of properties based on address identifiers, usually by road name, house number, postal code.'],
        'de': ['Adressen', 'Lokalisierung von Grundstücken anhand von Adressdaten, in der Regel Straßenname, Hausnummer und Postleitzahl.'],
    },
    'af': {
        'en': [
            'Agricultural and aquaculture facilities',
            'Farming equipment and production facilities (including irrigation systems, greenhouses and stables).',
        ],
        'de': [
            'Landwirtschaftliche Anlagen und Aquakulturanlagen',
            'Landwirtschaftliche Anlagen und Produktionsstätten (einschließlich Bewässerungssystemen, Gewächshäusern und Ställen).',
        ],
    },
    'am': {
        'en': [
            'Area management/restriction/regulation zones and reporting units',
            'Areas managed, regulated or used for reporting at international, European, national, regional and local levels. Includes dumping sites, restricted areas around drinking water sources, nitrate-vulnerable zones, regulated fairways at sea or large inland waters, areas for the dumping of waste, noise restriction zones, prospecting and mining permit areas, river basin districts, relevant reporting units and coastal zone management areas.',
        ],
        'de': [
            'Bewirtschaftungsgebiete/Schutzgebiete/geregelte Gebiete und Berichterstattungseinheiten',
            'Auf internationaler, europäischer, nationaler, regionaler und lokaler Ebene bewirtschaftete, geregelte oder zu Zwecken der Berichterstattung herangezogene Gebiete. Dazu zählen Deponien, Trinkwasserschutzgebiete, nitratempfindliche Gebiete, geregelte Fahrwasser auf See oder auf großen Binnengewässern, Gebiete für die Abfallverklappung, Lärmschutzgebiete, für Exploration und Bergbau ausgewiesene Gebiete, Flussgebietseinheiten, entsprechende Berichterstattungseinheiten und Gebiete des Küstenzonenmanagements.',
        ],
    },
    'au': {
        'en': [
            'Administrative units',
            'Units of administration, dividing areas where Member States have and/or exercise jurisdictional rights, for local, regional and national governance, separated by administrative boundaries.',
        ],
        'de': [
            'Verwaltungseinheiten',
            'Lokale, regionale und nationale Verwaltungseinheiten, die die Gebiete abgrenzen, in denen die Mitgliedstaaten Hoheitsbefugnisse haben und/oder ausüben und die durch Verwaltungsgrenzen voneinander getrennt sind.',
        ],
    },
    'br': {
        'en': ['Bio-geographical regions', 'Areas of relatively homogeneous ecological conditions with common characteristics.'],
        'de': ['Biogeografische Regionen', 'Gebiete mit relativ homogenen ökologischen Bedingungen und gemeinsamen Merkmalen.'],
    },
    'bu': {'en': ['Buildings', 'Geographical location of buildings.'], 'de': ['Gebäude', 'Geografischer Standort von Gebäuden.']},
    'cp': {
        'en': ['Cadastral parcels', 'Areas defined by cadastral registers or equivalent.'],
        'de': ['Flurstücke/Grundstücke (Katasterparzellen)', 'Gebiete, die anhand des Grundbuchs oder gleichwertiger Verzeichnisse bestimmt werden.'],
    },
    'ef': {
        'en': [
            'Environmental monitoring facilities',
            'Location and operation of environmental monitoring facilities includes observation and measurement of emissions, of the state of environmental media and of other ecosystem parameters (biodiversity, ecological conditions of vegetation, etc.) by or on behalf of public authorities.',
        ],
        'de': [
            'Umweltüberwachung',
            'Standort und Betrieb von Umweltüberwachungseinrichtungen einschließlich Beobachtung und Messung von Schadstoffen, des Zustands von Umweltmedien und anderen Parametern des Ökosystems (Artenvielfalt, ökologischer Zustand der Vegetation usw.) durch oder im Auftrag von öffentlichen Behörden.',
        ],
    },
    'el': {
        'en': ['Elevation', 'Digital elevation models for land, ice and ocean surface. Includes terrestrial elevation, bathymetry and shoreline.'],
        'de': ['Höhe', 'Digitale Höhenmodelle für Land-, Eis- und Meeresflächen. Dazu gehören Geländemodell, Tiefenmessung und Küstenlinie.'],
    },
    'er': {
        'en': [
            'Energy resources',
            'Energy resources including hydrocarbons, hydropower, bio-energy, solar, wind, etc., where relevant including depth/height information on the extent of the resource.',
        ],
        'de': [
            'Energiequellen',
            'Energiequellen wie Kohlenwasserstoffe, Wasserkraft, Bioenergie, Sonnen- und Windenergie usw., gegebenenfalls mit Tiefen- bzw. Höhenangaben zur Ausdehnung der Energiequelle.',
        ],
    },
    'ge': {
        'en': ['Geology', 'Geology characterised according to composition and structure. Includes bedrock, aquifers and geomorphology.'],
        'de': [
            'Geologie',
            'Geologische Beschreibung anhand von Zusammensetzung und Struktur. Dies umfasst auch Grundgestein, Grundwasserleiter und Geomorphologie.',
        ],
    },
    'gg': {
        'en': [
            'Geographical grid systems',
            'Harmonised multi-resolution grid with a common point of origin and standardised location and size of grid cells.',
        ],
        'de': [
            'Geografische Gittersysteme',
            'Harmonisiertes Gittersystem mit Mehrfachauflösung, gemeinsamem Ursprungspunkt und standardisierter Lokalisierung und Größe der Gitterzellen.',
        ],
    },
    'gn': {
        'en': [
            'Geographical names',
            'Names of areas, regions, localities, cities, suburbs, towns or settlements, or any geographical or topographical feature of public or historical interest.',
        ],
        'de': [
            'Geografische Bezeichnungen',
            'Namen von Gebieten, Regionen, Orten, Großstädten, Vororten, Städten oder Siedlungen sowie jedes geografische oder topografische Merkmal von öffentlichem oder historischem Interesse.',
        ],
    },
    'hb': {
        'en': [
            'Habitats and biotopes',
            'Geographical areas characterised by specific ecological conditions, processes, structure, and (life support) functions that physically support the organisms that live there. Includes terrestrial and aquatic areas distinguished by geographical, abiotic and biotic features, whether entirely natural or semi-natural.',
        ],
        'de': [
            'Lebensräume und Biotope',
            'Geografische Gebiete mit spezifischen ökologischen Bedingungen, Prozessen, Strukturen und (lebensunterstützenden) Funktionen als physische Grundlage für dort lebende Organismen. Dies umfasst auch durch geografische, abiotische und biotische Merkmale gekennzeichnete natürliche oder naturnahe terrestrische und aquatische Gebiete.',
        ],
    },
    'hh': {
        'en': [
            'Human health and safety',
            'Geographical distribution of dominance of pathologies (allergies, cancers, respiratory diseases, etc.), information indicating the effect on health (biomarkers, decline of fertility, epidemics) or well-being of humans (fatigue, stress, etc.) linked directly (air pollution, chemicals, depletion of the ozone layer, noise, etc.) or indirectly (food, genetically modified organisms, etc.) to the quality of the environment.',
        ],
        'de': [
            'Gesundheit und Sicherheit',
            'Geografische Verteilung verstärkt auftretender pathologischer Befunde (Allergien, Krebserkrankungen, Erkrankungen der Atemwege usw.), Informationen über Auswirkungen auf die Gesundheit (Biomarker, Rückgang der Fruchtbarkeit, Epidemien) oder auf das Wohlbefinden (Ermüdung, Stress usw.) der Menschen in unmittelbarem Zusammenhang mit der Umweltqualität (Luftverschmutzung, Chemikalien, Abbau der Ozonschicht, Lärm usw.) oder in mittelbarem Zusammenhang mit der Umweltqualität (Nahrung, genetisch veränderte Organismen usw.).',
        ],
    },
    'hy': {
        'en': [
            'Hydrography',
            'Hydrographic elements, including marine areas and all other water bodies and items related to them, including river basins and sub-basins. Where appropriate, according to the definitions set out in Directive 2000/60/EC of the European Parliament and of the Council of 23 October 2000 establishing a framework for Community action in the field of water policy (2) and in the form of networks.',
        ],
        'de': [
            'Gewässernetz',
            'Elemente des Gewässernetzes, einschließlich Meeresgebieten und allen sonstigen Wasserkörpern und hiermit verbundenen Teilsystemen, darunter Einzugsgebiete und Teileinzugsgebiete. Gegebenenfalls gemäß den Definitionen der Richtlinie 2000/60/EG des Europäischen Parlaments und des Rates vom 23. Oktober 2000 zur Schaffung eines Ordnungsrahmens für Maßnahmen der Gemeinschaft im Bereich der Wasserpolitik [2] und in Form von Netzen.',
        ],
    },
    'lc': {
        'en': [
            'Land cover',
            "Physical and biological cover of the earth's surface including artificial surfaces, agricultural areas, forests, (semi-)natural areas, wetlands, water bodies.",
        ],
        'de': [
            'Bodenbedeckung',
            'Physische und biologische Bedeckung der Erdoberfläche, einschließlich künstlicher Flächen, landwirtschaftlicher Flächen, Wäldern, natürlicher (naturnaher) Gebiete, Feuchtgebieten und Wasserkörpern.',
        ],
    },
    'lu': {
        'en': [
            'Land use',
            'Territory characterised according to its current and future planned functional dimension or socio-economic purpose (e.g. residential, industrial, commercial, agricultural, forestry, recreational).',
        ],
        'de': [
            'Bodennutzung',
            'Beschreibung von Gebieten anhand ihrer derzeitigen und geplanten künftigen Funktion oder ihres sozioökonomischen Zwecks (z. B. Wohn-, Industrie- oder Gewerbegebiete, land- oder forstwirtschaftliche Flächen, Freizeitgebiete).',
        ],
    },
    'mf': {
        'en': [
            'Meteorological geographical features',
            'Weather conditions and their measurements; precipitation, temperature, evapotranspiration, wind speed and direction.',
        ],
        'de': [
            'Meteorologisch-geografische Kennwerte',
            'Witterungsbedingungen und deren Messung; Niederschlag, Temperatur, Gesamtverdunstung (Evapotranspiration), Windgeschwindigkeit und Windrichtung.',
        ],
    },
    'mr': {
        'en': [
            'Mineral resources',
            'Mineral resources including metal ores, industrial minerals, etc., where relevant including depth/height information on the extent of the resource.',
        ],
        'de': [
            'Mineralische Bodenschätze',
            'Mineralische Bodenschätze wie Metallerze, Industrieminerale usw., gegebenenfalls mit Tiefen- bzw. Höhenangaben zur Ausdehnung der Bodenschätze.',
        ],
    },
    'nz': {
        'en': [
            'Natural risk zones',
            'Vulnerable areas characterised according to natural hazards (all atmospheric, hydrologic, seismic, volcanic and wildfire phenomena that, because of their location, severity, and frequency, have the potential to seriously affect society), e.g. floods, landslides and subsidence, avalanches, forest fires, earthquakes, volcanic eruptions.',
        ],
        'de': [
            'Gebiete mit naturbedingten Risiken',
            'Gefährdete Gebiete, eingestuft nach naturbedingten Risiken (sämtliche atmosphärischen, hydrologischen, seismischen, vulkanischen Phänomene sowie Naturfeuer, die aufgrund ihres örtlichen Auftretens sowie ihrer Schwere und Häufigkeit signifikante Auswirkungen auf die Gesellschaft haben können), z. B. Überschwemmungen, Erdrutsche und Bodensenkungen, Lawinen, Waldbrände, Erdbeben oder Vulkanausbrüche.',
        ],
    },
    'of': {
        'en': ['Oceanographic geographical features', 'Physical conditions of oceans (currents, salinity, wave heights, etc.).'],
        'de': ['Ozeanografisch-geografische Kennwerte', 'Physikalische Bedingungen der Ozeane (Strömungsverhältnisse, Salinität, Wellenhöhe usw.).'],
    },
    'oi': {
        'en': ['Orthoimagery', "Geo-referenced image data of the Earth's surface, from either satellite or airborne sensors."],
        'de': ['Orthofotografie', 'Georeferenzierte Bilddaten der Erdoberfläche von satelliten- oder luftfahrzeuggestützten Sensoren.'],
    },
    'pd': {
        'en': [
            'Population distribution — demography',
            'Geographical distribution of people, including population characteristics and activity levels, aggregated by grid, region, administrative unit or other analytical unit.',
        ],
        'de': [
            'Verteilung der Bevölkerung — Demografie',
            'Geografische Verteilung der Bevölkerung, einschließlich Bevölkerungsmerkmalen und Tätigkeitsebenen, zusammengefasst nach Gitter, Region, Verwaltungseinheit oder sonstigen analytischen Einheiten.',
        ],
    },
    'pf': {
        'en': [
            'Production and industrial facilities',
            'Industrial production sites, including installations covered by Council Directive 96/61/EC of 24 September 1996 concerning integrated pollution prevention and control (1) and water abstraction facilities, mining, storage sites.',
        ],
        'de': [
            'Produktions- und Industrieanlagen',
            'Standorte für industrielle Produktion, einschließlich durch die Richtlinie 96/61/EG des Rates vom 24. September 1996 über die integrierte Vermeidung und Verminderung der Umweltverschmutzung [1] erfasste Anlagen und Einrichtungen zur Wasserentnahme sowie Bergbau- und Lagerstandorte.',
        ],
    },
    'ps': {
        'en': [
            'Protected sites',
            "Area designated or managed within a framework of international, Community and Member States' legislation to achieve specific conservation objectives.",
        ],
        'de': [
            'Schutzgebiete',
            'Gebiete, die im Rahmen des internationalen und des gemeinschaftlichen Rechts sowie des Rechts der Mitgliedstaaten ausgewiesen sind oder verwaltet werden, um spezifische Erhaltungsziele zu erreichen.',
        ],
    },
    'rs': {
        'en': [
            'Coordinate reference systems',
            'Systems for uniquely referencing spatial information in space as a set of coordinates (x, y, z) and/or latitude and longitude and height, based on a geodetic horizontal and vertical datum.',
        ],
        'de': [
            'Koordinatenreferenzsysteme',
            'Systeme zur eindeutigen räumlichen Referenzierung von Geodaten anhand eines Koordinatensatzes (x, y, z) und/oder Angaben zu Breite, Länge und Höhe auf der Grundlage eines geodätischen horizontalen und vertikalen Datums.',
        ],
    },
    'sd': {
        'en': [
            'Species distribution',
            'Geographical distribution of occurrence of animal and plant species aggregated by grid, region, administrative unit or other analytical unit.',
        ],
        'de': [
            'Verteilung der Arten',
            'Geografische Verteilung des Auftretens von Tier- und Pflanzenarten, zusammengefasst in Gittern, Region, Verwaltungseinheit oder sonstigen analytischen Einheiten.',
        ],
    },
    'so': {
        'en': [
            'Soil',
            'Soils and subsoil characterised according to depth, texture, structure and content of particles and organic material, stoniness, erosion, where appropriate mean slope and anticipated water storage capacity.',
        ],
        'de': [
            'Boden',
            'Beschreibung von Boden und Unterboden anhand von Tiefe, Textur, Struktur und Gehalt an Teilchen sowie organischem Material, Steinigkeit, Erosion, gegebenenfalls durchschnittliches Gefälle und erwartete Wasserspeicherkapazität.',
        ],
    },
    'sr': {
        'en': [
            'Sea regions',
            'Physical conditions of seas and saline water bodies divided into regions and sub-regions with common characteristics.',
        ],
        'de': [
            'Meeresregionen',
            'Physikalische Bedingungen von Meeren und salzhaltigen Gewässern, aufgeteilt nach Regionen und Teilregionen mit gemeinsamen Merkmalen.',
        ],
    },
    'su': {
        'en': ['Statistical units', 'Units for dissemination or use of statistical information.'],
        'de': ['Statistische Einheiten', 'Einheiten für die Verbreitung oder Verwendung statistischer Daten.'],
    },
    'tn': {
        'en': [
            'Transport networks',
            'Road, rail, air and water transport networks and related infrastructure. Includes links between different networks. Also includes the trans-European transport network as defined in Decision No 1692/96/EC of the European Parliament and of the Council of 23 July 1996 on Community Guidelines for the development of the trans-European transport network (1) and future revisions of that Decision.',
        ],
        'de': [
            'Verkehrsnetze',
            'Verkehrsnetze und zugehörige Infrastruktureinrichtungen für Straßen-, Schienen- und Luftverkehr sowie Schifffahrt. Umfasst auch die Verbindungen zwischen den verschiedenen Netzen. Umfasst auch das transeuropäische Verkehrsnetz im Sinne der Entscheidung Nr. 1692/96/EG des Europäischen Parlaments und des Rates vom 23. Juli 1996 über gemeinschaftliche Leitlinien für den Aufbau eines transeuropäischen Verkehrsnetzes [1] und künftiger Überarbeitungen dieser Entscheidung.',
        ],
    },
    'us': {
        'en': [
            'Utility and governmental services',
            'Includes utility facilities such as sewage, waste management, energy supply and water supply, administrative and social governmental services such as public administrations, civil protection sites, schools and hospitals.',
        ],
        'de': [
            'Versorgungswirtschaft und staatliche Dienste',
            'Versorgungseinrichtungen wie Abwasser- und Abfallentsorgung, Energieversorgung und Wasserversorgung; staatliche Verwaltungs- und Sozialdienste wie öffentliche Verwaltung, Katastrophenschutz, Schulen und Krankenhäuser.',
        ],
    },
}


def theme_name(theme: str, language: str) -> str | None:
    """Retrieves the name of a theme in the specified language.

    Args:
        theme: The theme identifier.
        language: The language code.

    Returns:
        The name of the theme in the given language.

    Raises:
        KeyError: If the theme or language is not found in THEMES.
    """
    try:
        return THEMES[theme][language][0]
    except KeyError:
        pass


def theme_definition(theme: str, language: str) -> str | None:
    """Retrieves the definition of a theme in the specified language.

    Args:
        theme: The theme identifier.
        language: The language code.

    Returns:
        The definition of the theme in the given language.

    Raises:
        KeyError: If the theme or language is not found in THEMES.
    """
    try:
        return THEMES[theme][language][1]
    except KeyError:
        pass
