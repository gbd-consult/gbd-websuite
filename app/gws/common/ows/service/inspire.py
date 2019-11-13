"""Various inspire-related data."""

THEMES = (
    'au:AdministrativeBoundary',
    'au:AdministrativeUnit',
)

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