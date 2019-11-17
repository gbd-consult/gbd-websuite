"""Various inspire-related data."""

TAGS = {
    'au:AdministrativeBoundary': {},
    'au:AdministrativeUnit': {},
    'us-govserv:GovernmentalService': {
        'name': 'str',
        'address': 'str',
        'email': 'str',
        'fax': 'str',
        'phone': 'str',
        'serviceType': ('select', 'http://inspire.ec.europa.eu/codelist/ServiceTypeValue')
    }
}

NAMESPACES = {
    'ac-mf': (
        'https://inspire.ec.europa.eu/schemas/ac-mf/4.0/',
        'https://inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd'
    ),
    'act-core': (
        'https://inspire.ec.europa.eu/schemas/act-core/4.0/',
        'https://inspire.ec.europa.eu/schemas/act-core/4.0/ActivityComplex_Core.xsd'
    ),
    'ad': (
        'https://inspire.ec.europa.eu/schemas/ad/4.0/',
        'https://inspire.ec.europa.eu/schemas/ad/4.0/Addresses.xsd'
    ),
    'af': (
        'https://inspire.ec.europa.eu/schemas/af/4.0/',
        'https://inspire.ec.europa.eu/schemas/af/4.0/AgriculturalAndAquacultureFacilities.xsd'
    ),
    'am': (
        'https://inspire.ec.europa.eu/schemas/am/4.0/',
        'https://inspire.ec.europa.eu/schemas/am/4.0/AreaManagementRestrictionRegulationZone.xsd'
    ),
    'au': (
        'https://inspire.ec.europa.eu/schemas/au/4.0/',
        'https://inspire.ec.europa.eu/schemas/au/4.0/AdministrativeUnits.xsd'
    ),
    'base': (
        'https://inspire.ec.europa.eu/schemas/base/3.3/',
        'https://inspire.ec.europa.eu/schemas/base/3.3/BaseTypes.xsd'
    ),
    'base2': (
        'https://inspire.ec.europa.eu/schemas/base2/2.0/',
        'https://inspire.ec.europa.eu/schemas/base2/2.0/BaseTypes2.xsd'
    ),
    'br': (
        'https://inspire.ec.europa.eu/schemas/br/4.0/',
        'https://inspire.ec.europa.eu/schemas/br/4.0/Bio-geographicalRegions.xsd'
    ),
    'bu-base': (
        'https://inspire.ec.europa.eu/schemas/bu-base/4.0/',
        'https://inspire.ec.europa.eu/schemas/bu-base/4.0/BuildingsBase.xsd'
    ),
    'bu-core2d': (
        'https://inspire.ec.europa.eu/schemas/bu-core2d/4.0/',
        'https://inspire.ec.europa.eu/schemas/bu-core2d/4.0/BuildingsCore2D.xsd'
    ),
    'bu-core3d': (
        'https://inspire.ec.europa.eu/schemas/bu-core3d/4.0/',
        'https://inspire.ec.europa.eu/schemas/bu-core3d/4.0/BuildingsCore3D.xsd'
    ),
    'bu': (
        'https://inspire.ec.europa.eu/schemas/bu/0.0/',
        'https://inspire.ec.europa.eu/schemas/bu/0.0/Buildings.xsd'
    ),
    'common': (
        'https://inspire.ec.europa.eu/schemas/common/1.0/',
        'https://inspire.ec.europa.eu/schemas/common/1.0/common.xsd'
    ),
    'cp': (
        'https://inspire.ec.europa.eu/schemas/cp/4.0/',
        'https://inspire.ec.europa.eu/schemas/cp/4.0/CadastralParcels.xsd'
    ),
    'cvbase': (
        'https://inspire.ec.europa.eu/schemas/cvbase/2.0/',
        'https://inspire.ec.europa.eu/schemas/cvbase/2.0/CoverageBase.xsd'
    ),
    'cvgvp': (
        'https://inspire.ec.europa.eu/schemas/cvgvp/0.1/',
        'https://inspire.ec.europa.eu/schemas/cvgvp/0.1/CoverageGVP.xsd'
    ),
    'ef': (
        'https://inspire.ec.europa.eu/schemas/ef/4.0/',
        'https://inspire.ec.europa.eu/schemas/ef/4.0/EnvironmentalMonitoringFacilities.xsd'
    ),
    'el-bas': (
        'https://inspire.ec.europa.eu/schemas/el-bas/4.0/',
        'https://inspire.ec.europa.eu/schemas/el-bas/4.0/ElevationBaseTypes.xsd'
    ),
    'el-cov': (
        'https://inspire.ec.europa.eu/schemas/el-cov/4.0/',
        'https://inspire.ec.europa.eu/schemas/el-cov/4.0/ElevationGridCoverage.xsd'
    ),
    'el-tin': (
        'https://inspire.ec.europa.eu/schemas/el-tin/4.0/',
        'https://inspire.ec.europa.eu/schemas/el-tin/4.0/ElevationTin.xsd'
    ),
    'el-vec': (
        'https://inspire.ec.europa.eu/schemas/el-vec/4.0/',
        'https://inspire.ec.europa.eu/schemas/el-vec/4.0/ElevationVectorElements.xsd'
    ),
    'elu': (
        'https://inspire.ec.europa.eu/schemas/elu/4.0/',
        'https://inspire.ec.europa.eu/schemas/elu/4.0/ExistingLandUse.xsd'
    ),
    'er-b': (
        'https://inspire.ec.europa.eu/schemas/er-b/4.0/',
        'https://inspire.ec.europa.eu/schemas/er-b/4.0/EnergyResourcesBase.xsd'
    ),
    'er-c': (
        'https://inspire.ec.europa.eu/schemas/er-c/4.0/',
        'https://inspire.ec.europa.eu/schemas/er-c/4.0/EnergyResourcesCoverage.xsd'
    ),
    'er-v': (
        'https://inspire.ec.europa.eu/schemas/er-v/4.0/',
        'https://inspire.ec.europa.eu/schemas/er-v/4.0/EnergyResourcesVector.xsd'
    ),
    'er': (
        'https://inspire.ec.europa.eu/schemas/er/0.0/',
        'https://inspire.ec.europa.eu/schemas/er/0.0/EnergyResources.xsd'
    ),
    'gaz': (
        'https://inspire.ec.europa.eu/schemas/gaz/3.2/',
        'https://inspire.ec.europa.eu/schemas/gaz/3.2/Gazetteer.xsd'
    ),
    'ge-core': (
        'https://inspire.ec.europa.eu/schemas/ge-core/4.0/',
        'https://inspire.ec.europa.eu/schemas/ge-core/4.0/GeologyCore.xsd'
    ),
    'ge': (
        'https://inspire.ec.europa.eu/schemas/ge/0.0/',
        'https://inspire.ec.europa.eu/schemas/ge/0.0/Geology.xsd'
    ),
    'ge_gp': (
        'https://inspire.ec.europa.eu/schemas/ge_gp/4.0/',
        'https://inspire.ec.europa.eu/schemas/ge_gp/4.0/GeophysicsCore.xsd'
    ),
    'ge_hg': (
        'https://inspire.ec.europa.eu/schemas/ge_hg/4.0/',
        'https://inspire.ec.europa.eu/schemas/ge_hg/4.0/HydrogeologyCore.xsd'
    ),
    'gelu': (
        'https://inspire.ec.europa.eu/schemas/gelu/4.0/',
        'https://inspire.ec.europa.eu/schemas/gelu/4.0/GriddedExistingLandUse.xsd'
    ),
    'geoportal': (
        'https://inspire.ec.europa.eu/schemas/geoportal/1.0/',
        'https://inspire.ec.europa.eu/schemas/geoportal/1.0/geoportal.xsd'
    ),
    'gn': (
        'https://inspire.ec.europa.eu/schemas/gn/4.0/',
        'https://inspire.ec.europa.eu/schemas/gn/4.0/GeographicalNames.xsd'
    ),
    'hb': (
        'https://inspire.ec.europa.eu/schemas/hb/4.0/',
        'https://inspire.ec.europa.eu/schemas/hb/4.0/HabitatsAndBiotopes.xsd'
    ),
    'hh': (
        'https://inspire.ec.europa.eu/schemas/hh/4.0/',
        'https://inspire.ec.europa.eu/schemas/hh/4.0/HumanHealth.xsd'
    ),
    'hy-n': (
        'https://inspire.ec.europa.eu/schemas/hy-n/4.0/',
        'https://inspire.ec.europa.eu/schemas/hy-n/4.0/HydroNetwork.xsd'
    ),
    'hy-p': (
        'https://inspire.ec.europa.eu/schemas/hy-p/4.0/',
        'https://inspire.ec.europa.eu/schemas/hy-p/4.0/HydroPhysicalWaters.xsd'
    ),
    'hy': (
        'https://inspire.ec.europa.eu/schemas/hy/4.0/',
        'https://inspire.ec.europa.eu/schemas/hy/4.0/HydroBase.xsd'
    ),
    'inspire_dls': (
        'https://inspire.ec.europa.eu/schemas/inspire_dls/1.0/',
        'https://inspire.ec.europa.eu/schemas/inspire_dls/1.0/inspire_dls.xsd'
    ),
    'inspire_ds': (
        'https://inspire.ec.europa.eu/schemas/inspire_ds/1.0/',
        'https://inspire.ec.europa.eu/schemas/inspire_ds/1.0/inspire_ds.xsd'
    ),
    'inspire_vs': (
        'https://inspire.ec.europa.eu/schemas/inspire_vs/1.0/',
        'https://inspire.ec.europa.eu/schemas/inspire_vs/1.0/inspire_vs.xsd'
    ),
    'inspire_vs_ows11': (
        'https://inspire.ec.europa.eu/schemas/inspire_vs_ows11/1.0/',
        'https://inspire.ec.europa.eu/schemas/inspire_vs_ows11/1.0/inspire_vs_ows_11.xsd'
    ),
    'lc': (
        'https://inspire.ec.europa.eu/schemas/lc/0.0/',
        'https://inspire.ec.europa.eu/schemas/lc/0.0/LandCover.xsd'
    ),
    'lcn': (
        'https://inspire.ec.europa.eu/schemas/lcn/4.0/',
        'https://inspire.ec.europa.eu/schemas/lcn/4.0/LandCoverNomenclature.xsd'
    ),
    'lcr': (
        'https://inspire.ec.europa.eu/schemas/lcr/4.0/',
        'https://inspire.ec.europa.eu/schemas/lcr/4.0/LandCoverRaster.xsd'
    ),
    'lcv': (
        'https://inspire.ec.europa.eu/schemas/lcv/4.0/',
        'https://inspire.ec.europa.eu/schemas/lcv/4.0/LandCoverVector.xsd'
    ),
    'lunom': (
        'https://inspire.ec.europa.eu/schemas/lunom/4.0/',
        'https://inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd'
    ),
    'mr-core': (
        'https://inspire.ec.europa.eu/schemas/mr-core/4.0/',
        'https://inspire.ec.europa.eu/schemas/mr-core/4.0/MineralResourcesCore.xsd'
    ),
    'mu': (
        'https://inspire.ec.europa.eu/schemas/mu/3.0rc3/',
        'https://inspire.ec.europa.eu/schemas/mu/3.0rc3/MaritimeUnits.xsd'
    ),
    'net': (
        'https://inspire.ec.europa.eu/schemas/net/4.0/',
        'https://inspire.ec.europa.eu/schemas/net/4.0/Network.xsd'
    ),
    'nz-core': (
        'https://inspire.ec.europa.eu/schemas/nz-core/4.0/',
        'https://inspire.ec.europa.eu/schemas/nz-core/4.0/NaturalRiskZonesCore.xsd'
    ),
    'nz': (
        'https://inspire.ec.europa.eu/schemas/nz/0.0/',
        'https://inspire.ec.europa.eu/schemas/nz/0.0/NaturalRiskZones.xsd'
    ),
    'of': (
        'https://inspire.ec.europa.eu/schemas/of/4.0/',
        'https://inspire.ec.europa.eu/schemas/of/4.0/OceanFeatures.xsd'
    ),
    'oi': (
        'https://inspire.ec.europa.eu/schemas/oi/4.0/',
        'https://inspire.ec.europa.eu/schemas/oi/4.0/Orthoimagery.xsd'
    ),
    'omop': (
        'https://inspire.ec.europa.eu/schemas/omop/3.0/',
        'https://inspire.ec.europa.eu/schemas/omop/3.0/ObservableProperties.xsd'
    ),
    'omor': (
        'https://inspire.ec.europa.eu/schemas/omor/3.0/',
        'https://inspire.ec.europa.eu/schemas/omor/3.0/ObservationReferences.xsd'
    ),
    'ompr': (
        'https://inspire.ec.europa.eu/schemas/ompr/3.0/',
        'https://inspire.ec.europa.eu/schemas/ompr/3.0/Processes.xsd'
    ),
    'omso': (
        'https://inspire.ec.europa.eu/schemas/omso/3.0/',
        'https://inspire.ec.europa.eu/schemas/omso/3.0/SpecialisedObservations.xsd'
    ),
    'pd': (
        'https://inspire.ec.europa.eu/schemas/pd/4.0/',
        'https://inspire.ec.europa.eu/schemas/pd/4.0/PopulationDistributionDemography.xsd'
    ),
    'pf': (
        'https://inspire.ec.europa.eu/schemas/pf/4.0/',
        'https://inspire.ec.europa.eu/schemas/pf/4.0/ProductionAndIndustrialFacilities.xsd'
    ),
    'plu': (
        'https://inspire.ec.europa.eu/schemas/plu/4.0/',
        'https://inspire.ec.europa.eu/schemas/plu/4.0/PlannedLandUse.xsd'
    ),
    'ps': (
        'https://inspire.ec.europa.eu/schemas/ps/4.0/',
        'https://inspire.ec.europa.eu/schemas/ps/4.0/ProtectedSites.xsd'
    ),
    'sd': (
        'https://inspire.ec.europa.eu/schemas/sd/4.0/',
        'https://inspire.ec.europa.eu/schemas/sd/4.0/SpeciesDistribution.xsd'
    ),
    'selu': (
        'https://inspire.ec.europa.eu/schemas/selu/4.0/',
        'https://inspire.ec.europa.eu/schemas/selu/4.0/SampledExistingLandUse.xsd'
    ),
    'so': (
        'https://inspire.ec.europa.eu/schemas/so/4.0/',
        'https://inspire.ec.europa.eu/schemas/so/4.0/Soil.xsd'
    ),
    'sr': (
        'https://inspire.ec.europa.eu/schemas/sr/4.0/',
        'https://inspire.ec.europa.eu/schemas/sr/4.0/SeaRegions.xsd'
    ),
    'su-core': (
        'https://inspire.ec.europa.eu/schemas/su-core/4.0/',
        'https://inspire.ec.europa.eu/schemas/su-core/4.0/StatisticalUnitCore.xsd'
    ),
    'su-grid': (
        'https://inspire.ec.europa.eu/schemas/su-grid/4.0/',
        'https://inspire.ec.europa.eu/schemas/su-grid/4.0/StatisticalUnitGrid.xsd'
    ),
    'su-vector': (
        'https://inspire.ec.europa.eu/schemas/su-vector/4.0/',
        'https://inspire.ec.europa.eu/schemas/su-vector/4.0/StatisticalUnitVector.xsd'
    ),
    'su': (
        'https://inspire.ec.europa.eu/schemas/su/0.0/',
        'https://inspire.ec.europa.eu/schemas/su/0.0/StatisticalUnits.xsd'
    ),
    'tn-a': (
        'https://inspire.ec.europa.eu/schemas/tn-a/4.0/',
        'https://inspire.ec.europa.eu/schemas/tn-a/4.0/AirTransportNetwork.xsd'
    ),
    'tn-c': (
        'https://inspire.ec.europa.eu/schemas/tn-c/4.0/',
        'https://inspire.ec.europa.eu/schemas/tn-c/4.0/CableTransportNetwork.xsd'
    ),
    'tn-ra': (
        'https://inspire.ec.europa.eu/schemas/tn-ra/4.0/',
        'https://inspire.ec.europa.eu/schemas/tn-ra/4.0/RailwayTransportNetwork.xsd'
    ),
    'tn-ro': (
        'https://inspire.ec.europa.eu/schemas/tn-ro/4.0/',
        'https://inspire.ec.europa.eu/schemas/tn-ro/4.0/RoadTransportNetwork.xsd'
    ),
    'tn-w': (
        'https://inspire.ec.europa.eu/schemas/tn-w/4.0/',
        'https://inspire.ec.europa.eu/schemas/tn-w/4.0/WaterTransportNetwork.xsd'
    ),
    'tn': (
        'https://inspire.ec.europa.eu/schemas/tn/4.0/',
        'https://inspire.ec.europa.eu/schemas/tn/4.0/CommonTransportElements.xsd'
    ),
    'ugs': (
        'https://inspire.ec.europa.eu/schemas/ugs/0.0/',
        'https://inspire.ec.europa.eu/schemas/ugs/0.0/UtilityAndGovernmentalServices.xsd'
    ),
    'us-emf': (
        'https://inspire.ec.europa.eu/schemas/us-emf/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-emf/4.0/EnvironmentalManagementFacilities.xsd'
    ),
    'us-govserv': (
        'https://inspire.ec.europa.eu/schemas/us-govserv/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-govserv/4.0/GovernmentalServices.xsd'
    ),
    'us-net-common': (
        'https://inspire.ec.europa.eu/schemas/us-net-common/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-net-common/4.0/UtilityNetworksCommon.xsd'
    ),
    'us-net-el': (
        'https://inspire.ec.europa.eu/schemas/us-net-el/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-net-el/4.0/ElectricityNetwork.xsd'
    ),
    'us-net-ogc': (
        'https://inspire.ec.europa.eu/schemas/us-net-ogc/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-net-ogc/4.0/OilGasChemicalsNetwork.xsd'
    ),
    'us-net-sw': (
        'https://inspire.ec.europa.eu/schemas/us-net-sw/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-net-sw/4.0/SewerNetwork.xsd'
    ),
    'us-net-tc': (
        'https://inspire.ec.europa.eu/schemas/us-net-tc/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-net-tc/4.0/TelecommunicationsNetwork.xsd'
    ),
    'us-net-th': (
        'https://inspire.ec.europa.eu/schemas/us-net-th/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-net-th/4.0/ThermalNetwork.xsd'
    ),
    'us-net-wa': (
        'https://inspire.ec.europa.eu/schemas/us-net-wa/4.0/',
        'https://inspire.ec.europa.eu/schemas/us-net-wa/4.0/WaterNetwork.xsd'
    ),
    'wfd': (
        'https://inspire.ec.europa.eu/schemas/wfd/0.0/',
        'https://inspire.ec.europa.eu/schemas/wfd/0.0/WaterFrameworkDirective.xsd'
    ),
}

NAMESPACES['inspire_common'] = NAMESPACES['common']
NAMESPACES['lu'] = NAMESPACES['lunom']
NAMESPACES['ac'] = NAMESPACES['ac-mf']
NAMESPACES['mf'] = NAMESPACES['ac-mf']

# https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32007L0002&from=EN
# https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32007L0002&from=EN

THEMES = {
    "ac": {
        "en": [
            "Atmospheric conditions",
            "Physical conditions in the atmosphere. Includes spatial data based on measurements, on models or on a combination thereof and includes measurement locations."
        ],
        "de": [
            "Atmosphärische Bedingungen",
            "Physikalische Bedingungen in der Atmosphäre. Dazu zählen Geodaten auf der Grundlage von Messungen, Modellen oder einer Kombination aus beiden sowie Angabe der Messstandorte."
        ]
    },
    "ad": {
        "en": [
            "Addresses",
            "Location of properties based on address identifiers, usually by road name, house number, postal code."
        ],
        "de": [
            "Adressen",
            "Lokalisierung von Grundstücken anhand von Adressdaten, in der Regel Straßenname, Hausnummer und Postleitzahl."
        ]
    },
    "af": {
        "en": [
            "Agricultural and aquaculture facilities",
            "Farming equipment and production facilities (including irrigation systems, greenhouses and stables)."
        ],
        "de": [
            "Landwirtschaftliche Anlagen und Aquakulturanlagen",
            "Landwirtschaftliche Anlagen und Produktionsstätten (einschließlich Bewässerungssystemen, Gewächshäusern und Ställen)."
        ]
    },
    "am": {
        "en": [
            "Area management/restriction/regulation zones and reporting units",
            "Areas managed, regulated or used for reporting at international, European, national, regional and local levels. Includes dumping sites, restricted areas around drinking water sources, nitrate-vulnerable zones, regulated fairways at sea or large inland waters, areas for the dumping of waste, noise restriction zones, prospecting and mining permit areas, river basin districts, relevant reporting units and coastal zone management areas."
        ],
        "de": [
            "Bewirtschaftungsgebiete/Schutzgebiete/geregelte Gebiete und Berichterstattungseinheiten",
            "Auf internationaler, europäischer, nationaler, regionaler und lokaler Ebene bewirtschaftete, geregelte oder zu Zwecken der Berichterstattung herangezogene Gebiete. Dazu zählen Deponien, Trinkwasserschutzgebiete, nitratempfindliche Gebiete, geregelte Fahrwasser auf See oder auf großen Binnengewässern, Gebiete für die Abfallverklappung, Lärmschutzgebiete, für Exploration und Bergbau ausgewiesene Gebiete, Flussgebietseinheiten, entsprechende Berichterstattungseinheiten und Gebiete des Küstenzonenmanagements."
        ]
    },
    "au": {
        "en": [
            "Administrative units",
            "Units of administration, dividing areas where Member States have and/or exercise jurisdictional rights, for local, regional and national governance, separated by administrative boundaries."
        ],
        "de": [
            "Verwaltungseinheiten",
            "Lokale, regionale und nationale Verwaltungseinheiten, die die Gebiete abgrenzen, in denen die Mitgliedstaaten Hoheitsbefugnisse haben und/oder ausüben und die durch Verwaltungsgrenzen voneinander getrennt sind."
        ]
    },
    "br": {
        "en": [
            "Bio-geographical regions",
            "Areas of relatively homogeneous ecological conditions with common characteristics."
        ],
        "de": [
            "Biogeografische Regionen",
            "Gebiete mit relativ homogenen ökologischen Bedingungen und gemeinsamen Merkmalen."
        ]
    },
    "bu": {
        "en": [
            "Buildings",
            "Geographical location of buildings."
        ],
        "de": [
            "Gebäude",
            "Geografischer Standort von Gebäuden."
        ]
    },
    "cp": {
        "en": [
            "Cadastral parcels",
            "Areas defined by cadastral registers or equivalent."
        ],
        "de": [
            "Flurstücke/Grundstücke (Katasterparzellen)",
            "Gebiete, die anhand des Grundbuchs oder gleichwertiger Verzeichnisse bestimmt werden."
        ]
    },
    "ef": {
        "en": [
            "Environmental monitoring facilities",
            "Location and operation of environmental monitoring facilities includes observation and measurement of emissions, of the state of environmental media and of other ecosystem parameters (biodiversity, ecological conditions of vegetation, etc.) by or on behalf of public authorities."
        ],
        "de": [
            "Umweltüberwachung",
            "Standort und Betrieb von Umweltüberwachungseinrichtungen einschließlich Beobachtung und Messung von Schadstoffen, des Zustands von Umweltmedien und anderen Parametern des Ökosystems (Artenvielfalt, ökologischer Zustand der Vegetation usw.) durch oder im Auftrag von öffentlichen Behörden."
        ]
    },
    "el": {
        "en": [
            "Elevation",
            "Digital elevation models for land, ice and ocean surface. Includes terrestrial elevation, bathymetry and shoreline."
        ],
        "de": [
            "Höhe",
            "Digitale Höhenmodelle für Land-, Eis- und Meeresflächen. Dazu gehören Geländemodell, Tiefenmessung und Küstenlinie."
        ]
    },
    "er": {
        "en": [
            "Energy resources",
            "Energy resources including hydrocarbons, hydropower, bio-energy, solar, wind, etc., where relevant including depth/height information on the extent of the resource."
        ],
        "de": [
            "Energiequellen",
            "Energiequellen wie Kohlenwasserstoffe, Wasserkraft, Bioenergie, Sonnen- und Windenergie usw., gegebenenfalls mit Tiefen- bzw. Höhenangaben zur Ausdehnung der Energiequelle."
        ]
    },
    "ge": {
        "en": [
            "Geology",
            "Geology characterised according to composition and structure. Includes bedrock, aquifers and geomorphology."
        ],
        "de": [
            "Geologie",
            "Geologische Beschreibung anhand von Zusammensetzung und Struktur. Dies umfasst auch Grundgestein, Grundwasserleiter und Geomorphologie."
        ]
    },
    "gg": {
        "en": [
            "Geographical grid systems",
            "Harmonised multi-resolution grid with a common point of origin and standardised location and size of grid cells."
        ],
        "de": [
            "Geografische Gittersysteme",
            "Harmonisiertes Gittersystem mit Mehrfachauflösung, gemeinsamem Ursprungspunkt und standardisierter Lokalisierung und Größe der Gitterzellen."
        ]
    },
    "gn": {
        "en": [
            "Geographical names",
            "Names of areas, regions, localities, cities, suburbs, towns or settlements, or any geographical or topographical feature of public or historical interest."
        ],
        "de": [
            "Geografische Bezeichnungen",
            "Namen von Gebieten, Regionen, Orten, Großstädten, Vororten, Städten oder Siedlungen sowie jedes geografische oder topografische Merkmal von öffentlichem oder historischem Interesse."
        ]
    },
    "hb": {
        "en": [
            "Habitats and biotopes",
            "Geographical areas characterised by specific ecological conditions, processes, structure, and (life support) functions that physically support the organisms that live there. Includes terrestrial and aquatic areas distinguished by geographical, abiotic and biotic features, whether entirely natural or semi-natural."
        ],
        "de": [
            "Lebensräume und Biotope",
            "Geografische Gebiete mit spezifischen ökologischen Bedingungen, Prozessen, Strukturen und (lebensunterstützenden) Funktionen als physische Grundlage für dort lebende Organismen. Dies umfasst auch durch geografische, abiotische und biotische Merkmale gekennzeichnete natürliche oder naturnahe terrestrische und aquatische Gebiete."
        ]
    },
    "hh": {
        "en": [
            "Human health and safety",
            "Geographical distribution of dominance of pathologies (allergies, cancers, respiratory diseases, etc.), information indicating the effect on health (biomarkers, decline of fertility, epidemics) or well-being of humans (fatigue, stress, etc.) linked directly (air pollution, chemicals, depletion of the ozone layer, noise, etc.) or indirectly (food, genetically modified organisms, etc.) to the quality of the environment."
        ],
        "de": [
            "Gesundheit und Sicherheit",
            "Geografische Verteilung verstärkt auftretender pathologischer Befunde (Allergien, Krebserkrankungen, Erkrankungen der Atemwege usw.), Informationen über Auswirkungen auf die Gesundheit (Biomarker, Rückgang der Fruchtbarkeit, Epidemien) oder auf das Wohlbefinden (Ermüdung, Stress usw.) der Menschen in unmittelbarem Zusammenhang mit der Umweltqualität (Luftverschmutzung, Chemikalien, Abbau der Ozonschicht, Lärm usw.) oder in mittelbarem Zusammenhang mit der Umweltqualität (Nahrung, genetisch veränderte Organismen usw.)."
        ]
    },
    "hy": {
        "en": [
            "Hydrography",
            "Hydrographic elements, including marine areas and all other water bodies and items related to them, including river basins and sub-basins. Where appropriate, according to the definitions set out in Directive 2000/60/EC of the European Parliament and of the Council of 23 October 2000 establishing a framework for Community action in the field of water policy (2) and in the form of networks."
        ],
        "de": [
            "Gewässernetz",
            "Elemente des Gewässernetzes, einschließlich Meeresgebieten und allen sonstigen Wasserkörpern und hiermit verbundenen Teilsystemen, darunter Einzugsgebiete und Teileinzugsgebiete. Gegebenenfalls gemäß den Definitionen der Richtlinie 2000/60/EG des Europäischen Parlaments und des Rates vom 23. Oktober 2000 zur Schaffung eines Ordnungsrahmens für Maßnahmen der Gemeinschaft im Bereich der Wasserpolitik (2) und in Form von Netzen."
        ]
    },
    "lc": {
        "en": [
            "Land cover",
            "Physical and biological cover of the earth's surface including artificial surfaces, agricultural areas, forests, (semi-)natural areas, wetlands, water bodies."
        ],
        "de": [
            "Bodenbedeckung",
            "Physische und biologische Bedeckung der Erdoberfläche, einschließlich künstlicher Flächen, landwirtschaftlicher Flächen, Wäldern, natürlicher (naturnaher) Gebiete, Feuchtgebieten und Wasserkörpern."
        ]
    },
    "lu": {
        "en": [
            "Land use",
            "Territory characterised according to its current and future planned functional dimension or socio-economic purpose (e.g. residential, industrial, commercial, agricultural, forestry, recreational)."
        ],
        "de": [
            "Bodennutzung",
            "Beschreibung von Gebieten anhand ihrer derzeitigen und geplanten künftigen Funktion oder ihres sozioökonomischen Zwecks (z. B. Wohn-, Industrie- oder Gewerbegebiete, land- oder forstwirtschaftliche Flächen, Freizeitgebiete)."
        ]
    },
    "mf": {
        "en": [
            "Meteorological geographical features",
            "Weather conditions and their measurements; precipitation, temperature, evapotranspiration, wind speed and direction."
        ],
        "de": [
            "Meteorologisch-geografische Kennwerte",
            "Witterungsbedingungen und deren Messung; Niederschlag, Temperatur, Gesamtverdunstung (Evapotranspiration), Windgeschwindigkeit und Windrichtung."
        ]
    },
    "mr": {
        "en": [
            "Mineral resources",
            "Mineral resources including metal ores, industrial minerals, etc., where relevant including depth/height information on the extent of the resource."
        ],
        "de": [
            "Mineralische Bodenschätze",
            "Mineralische Bodenschätze wie Metallerze, Industrieminerale usw., gegebenenfalls mit Tiefen- bzw. Höhenangaben zur Ausdehnung der Bodenschätze."
        ]
    },
    "nz": {
        "en": [
            "Natural risk zones",
            "Vulnerable areas characterised according to natural hazards (all atmospheric, hydrologic, seismic, volcanic and wildfire phenomena that, because of their location, severity, and frequency, have the potential to seriously affect society), e.g. floods, landslides and subsidence, avalanches, forest fires, earthquakes, volcanic eruptions."
        ],
        "de": [
            "Gebiete mit naturbedingten Risiken",
            "Gefährdete Gebiete, eingestuft nach naturbedingten Risiken (sämtliche atmosphärischen, hydrologischen, seismischen, vulkanischen Phänomene sowie Naturfeuer, die aufgrund ihres örtlichen Auftretens sowie ihrer Schwere und Häufigkeit signifikante Auswirkungen auf die Gesellschaft haben können), z. B. Überschwemmungen, Erdrutsche und Bodensenkungen, Lawinen, Waldbrände, Erdbeben oder Vulkanausbrüche."
        ]
    },
    "of": {
        "en": [
            "Oceanographic geographical features",
            "Physical conditions of oceans (currents, salinity, wave heights, etc.)."
        ],
        "de": [
            "Ozeanografisch-geografische Kennwerte",
            "Physikalische Bedingungen der Ozeane (Strömungsverhältnisse, Salinität, Wellenhöhe usw.)."
        ]
    },
    "oi": {
        "en": [
            "Orthoimagery",
            "Geo-referenced image data of the Earth's surface, from either satellite or airborne sensors."
        ],
        "de": [
            "Orthofotografie",
            "Georeferenzierte Bilddaten der Erdoberfläche von satelliten- oder luftfahrzeuggestützten Sensoren."
        ]
    },
    "pd": {
        "en": [
            "Population distribution — demography",
            "Geographical distribution of people, including population characteristics and activity levels, aggregated by grid, region, administrative unit or other analytical unit."
        ],
        "de": [
            "Verteilung der Bevölkerung — Demografie",
            "Geografische Verteilung der Bevölkerung, einschließlich Bevölkerungsmerkmalen und Tätigkeitsebenen, zusammengefasst nach Gitter, Region, Verwaltungseinheit oder sonstigen analytischen Einheiten."
        ]
    },
    "pf": {
        "en": [
            "Production and industrial facilities",
            "Industrial production sites, including installations covered by Council Directive 96/61/EC of 24 September 1996 concerning integrated pollution prevention and control (1) and water abstraction facilities, mining, storage sites."
        ],
        "de": [
            "Produktions- und Industrieanlagen",
            "Standorte für industrielle Produktion, einschließlich durch die Richtlinie 96/61/EG des Rates vom 24. September 1996 über die integrierte Vermeidung und Verminderung der Umweltverschmutzung (1) erfasste Anlagen und Einrichtungen zur Wasserentnahme sowie Bergbau- und Lagerstandorte."
        ]
    },
    "ps": {
        "en": [
            "Protected sites",
            "Area designated or managed within a framework of international, Community and Member States' legislation to achieve specific conservation objectives."
        ],
        "de": [
            "Schutzgebiete",
            "Gebiete, die im Rahmen des internationalen und des gemeinschaftlichen Rechts sowie des Rechts der Mitgliedstaaten ausgewiesen sind oder verwaltet werden, um spezifische Erhaltungsziele zu erreichen."
        ]
    },
    "rs": {
        "en": [
            "Coordinate reference systems",
            "Systems for uniquely referencing spatial information in space as a set of coordinates (x, y, z) and/or latitude and longitude and height, based on a geodetic horizontal and vertical datum."
        ],
        "de": [
            "Koordinatenreferenzsysteme",
            "Systeme zur eindeutigen räumlichen Referenzierung von Geodaten anhand eines Koordinatensatzes (x, y, z) und/oder Angaben zu Breite, Länge und Höhe auf der Grundlage eines geodätischen horizontalen und vertikalen Datums."
        ]
    },
    "sd": {
        "en": [
            "Species distribution",
            "Geographical distribution of occurrence of animal and plant species aggregated by grid, region, administrative unit or other analytical unit."
        ],
        "de": [
            "Verteilung der Arten",
            "Geografische Verteilung des Auftretens von Tier- und Pflanzenarten, zusammengefasst in Gittern, Region, Verwaltungseinheit oder sonstigen analytischen Einheiten."
        ]
    },
    "so": {
        "en": [
            "Soil",
            "Soils and subsoil characterised according to depth, texture, structure and content of particles and organic material, stoniness, erosion, where appropriate mean slope and anticipated water storage capacity."
        ],
        "de": [
            "Boden",
            "Beschreibung von Boden und Unterboden anhand von Tiefe, Textur, Struktur und Gehalt an Teilchen sowie organischem Material, Steinigkeit, Erosion, gegebenenfalls durchschnittliches Gefälle und erwartete Wasserspeicherkapazität."
        ]
    },
    "sr": {
        "en": [
            "Sea regions",
            "Physical conditions of seas and saline water bodies divided into regions and sub-regions with common characteristics."
        ],
        "de": [
            "Meeresregionen",
            "Physikalische Bedingungen von Meeren und salzhaltigen Gewässern, aufgeteilt nach Regionen und Teilregionen mit gemeinsamen Merkmalen."
        ]
    },
    "su": {
        "en": [
            "Statistical units",
            "Units for dissemination or use of statistical information."
        ],
        "de": [
            "Statistische Einheiten",
            "Einheiten für die Verbreitung oder Verwendung statistischer Daten."
        ]
    },
    "tn": {
        "en": [
            "Transport networks",
            "Road, rail, air and water transport networks and related infrastructure. Includes links between different networks. Also includes the trans-European transport network as defined in Decision No 1692/96/EC of the European Parliament and of the Council of 23 July 1996 on Community Guidelines for the development of the trans-European transport network (1) and future revisions of that Decision."
        ],
        "de": [
            "Verkehrsnetze",
            "Verkehrsnetze und zugehörige Infrastruktureinrichtungen für Straßen-, Schienen- und Luftverkehr sowie Schifffahrt. Umfasst auch die Verbindungen zwischen den verschiedenen Netzen. Umfasst auch das transeuropäische Verkehrsnetz im Sinne der Entscheidung Nr. 1692/96/EG des Europäischen Parlaments und des Rates vom 23. Juli 1996 über gemeinschaftliche Leitlinien für den Aufbau eines transeuropäischen Verkehrsnetzes (1) und künftiger Überarbeitungen dieser Entscheidung."
        ]
    },
    "us": {
        "en": [
            "Utility and governmental services",
            "Includes utility facilities such as sewage, waste management, energy supply and water supply, administrative and social governmental services such as public administrations, civil protection sites, schools and hospitals."
        ],
        "de": [
            "Versorgungswirtschaft und staatliche Dienste",
            "Versorgungseinrichtungen wie Abwasser- und Abfallentsorgung, Energieversorgung und Wasserversorgung; staatliche Verwaltungs- und Sozialdienste wie öffentliche Verwaltung, Katastrophenschutz, Schulen und Krankenhäuser."
        ]
    }
}


def theme_name(theme, language):
    try:
        return THEMES[theme][language][0]
    except KeyError:
        pass


def theme_description(theme, language):
    try:
        return THEMES[theme][language][1]
    except KeyError:
        pass
