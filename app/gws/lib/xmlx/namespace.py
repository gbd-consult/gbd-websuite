# Namespaces used in built-in templates.

from . import error


def register(prefix, uri, schema_location=''):
    _namespaces.add(prefix, uri, schema_location)


def declarations(default_prefix=None, for_element=None, prefixes=None, with_schema_locations=True):
    return _namespaces.declarations(
        default_prefix,
        for_element,
        prefixes,
        with_schema_locations,
    )


def unqualify(name):
    if ':' in name:
        s = name.split(':')
        return s[-1]

    if '}' in name:
        s = name.split('}')
        return s[-1]

    return name


def qualify(name, fallback_prefix=None, generate_missing=True):
    if ':' in name:
        return name

    if '}' in name:
        s = name[1:].split('}')
        uri = s[0]
        pfx = _namespaces.prefix(uri, generate_missing=generate_missing) or fallback_prefix
        if not pfx:
            raise error.WriteError(f'unknown namespace uri {uri!r}')
        return pfx + ':' + name

    if fallback_prefix:
        return fallback_prefix + ':' + name

    return name


##

_KNOWN_NAMESPACES = [
    # (id, url, schema-location)

    # from ElementTree

    ('xml', 'http://www.w3.org/XML/1998/namespace', ''),
    ('html', 'http://www.w3.org/1999/xhtml', ''),
    ('wsdl', 'http://schemas.xmlsoap.org/wsdl/', ''),
    ('xs', 'http://www.w3.org/2001/XMLSchema', ''),

    # NB our templates use "xsd" and "xsi" for XMLSchema namespaces
    ('xsd', 'http://www.w3.org/2001/XMLSchema', ''),
    ('xsi', 'http://www.w3.org/2001/XMLSchema-instance', ''),
    ('xlink', 'http://www.w3.org/1999/xlink', 'https://www.w3.org/XML/2008/06/xlink.xsd'),
    ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns', ''),
    ('soap', 'http://www.w3.org/2003/05/soap-envelope', 'https://www.w3.org/2003/05/soap-envelope/'),

    # ogc

    ('csw', 'http://www.opengis.net/cat/csw/2.0.2', 'http://schemas.opengis.net/csw/2.0.2/csw.xsd'),
    ('dc', 'http://purl.org/dc/elements/1.1/', 'http://schemas.opengis.net/csw/2.0.2/rec-dcmes.xsd'),
    ('dcm', 'http://purl.org/dc/dcmitype/', 'http://dublincore.org/schemas/xmls/qdc/2008/02/11/dcmitype.xsd'),
    ('dct', 'http://purl.org/dc/terms/', 'http://schemas.opengis.net/csw/2.0.2/rec-dcterms.xsd'),
    ('fes', 'http://www.opengis.net/fes/2.0', 'http://schemas.opengis.net/filter/2.0/filterAll.xsd'),
    ('gco', 'http://www.isotc211.org/2005/gco', 'http://schemas.opengis.net/iso/19139/20070417/gco/gco.xsd'),
    ('gmd', 'http://www.isotc211.org/2005/gmd', 'http://schemas.opengis.net/csw/2.0.2/profiles/apiso/1.0.0/apiso.xsd'),
    ('gml', 'http://www.opengis.net/gml/3.2', 'http://schemas.opengis.net/gml/3.2.1/gml.xsd'),
    ('gmlcov', 'http://www.opengis.net/gmlcov/1.0', 'http://schemas.opengis.net/gmlcov/1.0/gmlcovAll.xsd'),
    ('gmx', 'http://www.isotc211.org/2005/gmx', 'http://schemas.opengis.net/iso/19139/20070417/gmx/gmx.xsd'),
    ('ogc', 'http://www.opengis.net/ogc', 'http://schemas.opengis.net/filter/1.1.0/filter.xsd'),
    ('ows', 'http://www.opengis.net/ows/1.1', 'http://schemas.opengis.net/ows/1.0.0/owsAll.xsd'),
    ('sld', 'http://www.opengis.net/sld', 'http://schemas.opengis.net/sld/1.1/sldAll.xsd'),
    ('srv', 'http://www.isotc211.org/2005/srv', 'http://schemas.opengis.net/iso/19139/20070417/srv/1.0/srv.xsd'),
    ('swe', 'http://www.opengis.net/swe/2.0', 'http://schemas.opengis.net/sweCommon/2.0/swe.xsd'),
    ('wcs', 'http://www.opengis.net/wcs/2.0', 'http://schemas.opengis.net/wcs/1.0.0/wcsAll.xsd'),
    ('wcscrs', 'http://www.opengis.net/wcs/crs/1.0', ''),
    ('wcsint', 'http://www.opengis.net/wcs/interpolation/1.0', ''),
    ('wcsscal', 'http://www.opengis.net/wcs/scaling/1.0', ''),
    ('wfs', 'http://www.opengis.net/wfs/2.0', 'http://schemas.opengis.net/wfs/2.0/wfs.xsd'),
    ('wms', 'http://www.opengis.net/wms', 'http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd'),
    ('wmts', 'http://www.opengis.net/wmts/1.0', 'http://schemas.opengis.net/wmts/1.0/wmts.xsd'),

    # inspire

    ('inspire_dls', 'http://inspire.ec.europa.eu/schemas/inspire_dls/1.0', 'http://inspire.ec.europa.eu/schemas/inspire_dls/1.0/inspire_dls.xsd'),
    ('inspire_ds', 'http://inspire.ec.europa.eu/schemas/inspire_ds/1.0', 'http://inspire.ec.europa.eu/schemas/inspire_ds/1.0/inspire_ds.xsd'),
    ('inspire_vs', 'http://inspire.ec.europa.eu/schemas/inspire_vs/1.0', 'http://inspire.ec.europa.eu/schemas/inspire_vs/1.0/inspire_vs.xsd'),
    ('inspire_vs_ows11', 'http://inspire.ec.europa.eu/schemas/inspire_vs_ows11/1.0', 'http://inspire.ec.europa.eu/schemas/inspire_vs_ows11/1.0/inspire_vs_ows_11.xsd'),
    ('inspire_common', 'http://inspire.ec.europa.eu/schemas/common/1.0', 'http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd'),

    # inspire themes

    ('ac-mf', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd'),
    ('ac', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd'),
    ('mf', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0', 'http://inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd'),
    ('act-core', 'http://inspire.ec.europa.eu/schemas/act-core/4.0', 'http://inspire.ec.europa.eu/schemas/act-core/4.0/ActivityComplex_Core.xsd'),
    ('ad', 'http://inspire.ec.europa.eu/schemas/ad/4.0', 'http://inspire.ec.europa.eu/schemas/ad/4.0/Addresses.xsd'),
    ('af', 'http://inspire.ec.europa.eu/schemas/af/4.0', 'http://inspire.ec.europa.eu/schemas/af/4.0/AgriculturalAndAquacultureFacilities.xsd'),
    ('am', 'http://inspire.ec.europa.eu/schemas/am/4.0', 'http://inspire.ec.europa.eu/schemas/am/4.0/AreaManagementRestrictionRegulationZone.xsd'),
    ('au', 'http://inspire.ec.europa.eu/schemas/au/4.0', 'http://inspire.ec.europa.eu/schemas/au/4.0/AdministrativeUnits.xsd'),
    ('base', 'http://inspire.ec.europa.eu/schemas/base/3.3', 'http://inspire.ec.europa.eu/schemas/base/3.3/BaseTypes.xsd'),
    ('base2', 'http://inspire.ec.europa.eu/schemas/base2/2.0', 'http://inspire.ec.europa.eu/schemas/base2/2.0/BaseTypes2.xsd'),
    ('br', 'http://inspire.ec.europa.eu/schemas/br/4.0', 'http://inspire.ec.europa.eu/schemas/br/4.0/Bio-geographicalRegions.xsd'),
    ('bu-base', 'http://inspire.ec.europa.eu/schemas/bu-base/4.0', 'http://inspire.ec.europa.eu/schemas/bu-base/4.0/BuildingsBase.xsd'),
    ('bu-core2d', 'http://inspire.ec.europa.eu/schemas/bu-core2d/4.0', 'http://inspire.ec.europa.eu/schemas/bu-core2d/4.0/BuildingsCore2D.xsd'),
    ('bu-core3d', 'http://inspire.ec.europa.eu/schemas/bu-core3d/4.0', 'http://inspire.ec.europa.eu/schemas/bu-core3d/4.0/BuildingsCore3D.xsd'),
    ('bu', 'http://inspire.ec.europa.eu/schemas/bu/0.0', 'http://inspire.ec.europa.eu/schemas/bu/0.0/Buildings.xsd'),
    ('cp', 'http://inspire.ec.europa.eu/schemas/cp/4.0', 'http://inspire.ec.europa.eu/schemas/cp/4.0/CadastralParcels.xsd'),
    ('cvbase', 'http://inspire.ec.europa.eu/schemas/cvbase/2.0', 'http://inspire.ec.europa.eu/schemas/cvbase/2.0/CoverageBase.xsd'),
    ('cvgvp', 'http://inspire.ec.europa.eu/schemas/cvgvp/0.1', 'http://inspire.ec.europa.eu/schemas/cvgvp/0.1/CoverageGVP.xsd'),
    ('ef', 'http://inspire.ec.europa.eu/schemas/ef/4.0', 'http://inspire.ec.europa.eu/schemas/ef/4.0/EnvironmentalMonitoringFacilities.xsd'),
    ('el-bas', 'http://inspire.ec.europa.eu/schemas/el-bas/4.0', 'http://inspire.ec.europa.eu/schemas/el-bas/4.0/ElevationBaseTypes.xsd'),
    ('el-cov', 'http://inspire.ec.europa.eu/schemas/el-cov/4.0', 'http://inspire.ec.europa.eu/schemas/el-cov/4.0/ElevationGridCoverage.xsd'),
    ('el-tin', 'http://inspire.ec.europa.eu/schemas/el-tin/4.0', 'http://inspire.ec.europa.eu/schemas/el-tin/4.0/ElevationTin.xsd'),
    ('el-vec', 'http://inspire.ec.europa.eu/schemas/el-vec/4.0', 'http://inspire.ec.europa.eu/schemas/el-vec/4.0/ElevationVectorElements.xsd'),
    ('elu', 'http://inspire.ec.europa.eu/schemas/elu/4.0', 'http://inspire.ec.europa.eu/schemas/elu/4.0/ExistingLandUse.xsd'),
    ('er-b', 'http://inspire.ec.europa.eu/schemas/er-b/4.0', 'http://inspire.ec.europa.eu/schemas/er-b/4.0/EnergyResourcesBase.xsd'),
    ('er-c', 'http://inspire.ec.europa.eu/schemas/er-c/4.0', 'http://inspire.ec.europa.eu/schemas/er-c/4.0/EnergyResourcesCoverage.xsd'),
    ('er-v', 'http://inspire.ec.europa.eu/schemas/er-v/4.0', 'http://inspire.ec.europa.eu/schemas/er-v/4.0/EnergyResourcesVector.xsd'),
    ('er', 'http://inspire.ec.europa.eu/schemas/er/0.0', 'http://inspire.ec.europa.eu/schemas/er/0.0/EnergyResources.xsd'),
    ('gaz', 'http://inspire.ec.europa.eu/schemas/gaz/3.2', 'http://inspire.ec.europa.eu/schemas/gaz/3.2/Gazetteer.xsd'),
    ('ge-core', 'http://inspire.ec.europa.eu/schemas/ge-core/4.0', 'http://inspire.ec.europa.eu/schemas/ge-core/4.0/GeologyCore.xsd'),
    ('ge', 'http://inspire.ec.europa.eu/schemas/ge/0.0', 'http://inspire.ec.europa.eu/schemas/ge/0.0/Geology.xsd'),
    ('ge_gp', 'http://inspire.ec.europa.eu/schemas/ge_gp/4.0', 'http://inspire.ec.europa.eu/schemas/ge_gp/4.0/GeophysicsCore.xsd'),
    ('ge_hg', 'http://inspire.ec.europa.eu/schemas/ge_hg/4.0', 'http://inspire.ec.europa.eu/schemas/ge_hg/4.0/HydrogeologyCore.xsd'),
    ('gelu', 'http://inspire.ec.europa.eu/schemas/gelu/4.0', 'http://inspire.ec.europa.eu/schemas/gelu/4.0/GriddedExistingLandUse.xsd'),
    ('geoportal', 'http://inspire.ec.europa.eu/schemas/geoportal/1.0', 'http://inspire.ec.europa.eu/schemas/geoportal/1.0/geoportal.xsd'),
    ('gn', 'http://inspire.ec.europa.eu/schemas/gn/4.0', 'http://inspire.ec.europa.eu/schemas/gn/4.0/GeographicalNames.xsd'),
    ('hb', 'http://inspire.ec.europa.eu/schemas/hb/4.0', 'http://inspire.ec.europa.eu/schemas/hb/4.0/HabitatsAndBiotopes.xsd'),
    ('hh', 'http://inspire.ec.europa.eu/schemas/hh/4.0', 'http://inspire.ec.europa.eu/schemas/hh/4.0/HumanHealth.xsd'),
    ('hy-n', 'http://inspire.ec.europa.eu/schemas/hy-n/4.0', 'http://inspire.ec.europa.eu/schemas/hy-n/4.0/HydroNetwork.xsd'),
    ('hy-p', 'http://inspire.ec.europa.eu/schemas/hy-p/4.0', 'http://inspire.ec.europa.eu/schemas/hy-p/4.0/HydroPhysicalWaters.xsd'),
    ('hy', 'http://inspire.ec.europa.eu/schemas/hy/4.0', 'http://inspire.ec.europa.eu/schemas/hy/4.0/HydroBase.xsd'),
    ('lc', 'http://inspire.ec.europa.eu/schemas/lc/0.0', 'http://inspire.ec.europa.eu/schemas/lc/0.0/LandCover.xsd'),
    ('lcn', 'http://inspire.ec.europa.eu/schemas/lcn/4.0', 'http://inspire.ec.europa.eu/schemas/lcn/4.0/LandCoverNomenclature.xsd'),
    ('lcr', 'http://inspire.ec.europa.eu/schemas/lcr/4.0', 'http://inspire.ec.europa.eu/schemas/lcr/4.0/LandCoverRaster.xsd'),
    ('lcv', 'http://inspire.ec.europa.eu/schemas/lcv/4.0', 'http://inspire.ec.europa.eu/schemas/lcv/4.0/LandCoverVector.xsd'),
    ('lu', 'http://inspire.ec.europa.eu/schemas/lunom/4.0', 'http://inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd'),
    ('lunom', 'http://inspire.ec.europa.eu/schemas/lunom/4.0', 'http://inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd'),
    ('mr-core', 'http://inspire.ec.europa.eu/schemas/mr-core/4.0', 'http://inspire.ec.europa.eu/schemas/mr-core/4.0/MineralResourcesCore.xsd'),
    ('mu', 'http://inspire.ec.europa.eu/schemas/mu/3.0rc3', 'http://inspire.ec.europa.eu/schemas/mu/3.0rc3/MaritimeUnits.xsd'),
    ('net', 'http://inspire.ec.europa.eu/schemas/net/4.0', 'http://inspire.ec.europa.eu/schemas/net/4.0/Network.xsd'),
    ('nz-core', 'http://inspire.ec.europa.eu/schemas/nz-core/4.0', 'http://inspire.ec.europa.eu/schemas/nz-core/4.0/NaturalRiskZonesCore.xsd'),
    ('nz', 'http://inspire.ec.europa.eu/schemas/nz/0.0', 'http://inspire.ec.europa.eu/schemas/nz/0.0/NaturalRiskZones.xsd'),
    ('of', 'http://inspire.ec.europa.eu/schemas/of/4.0', 'http://inspire.ec.europa.eu/schemas/of/4.0/OceanFeatures.xsd'),
    ('oi', 'http://inspire.ec.europa.eu/schemas/oi/4.0', 'http://inspire.ec.europa.eu/schemas/oi/4.0/Orthoimagery.xsd'),
    ('omop', 'http://inspire.ec.europa.eu/schemas/omop/3.0', 'http://inspire.ec.europa.eu/schemas/omop/3.0/ObservableProperties.xsd'),
    ('omor', 'http://inspire.ec.europa.eu/schemas/omor/3.0', 'http://inspire.ec.europa.eu/schemas/omor/3.0/ObservationReferences.xsd'),
    ('ompr', 'http://inspire.ec.europa.eu/schemas/ompr/3.0', 'http://inspire.ec.europa.eu/schemas/ompr/3.0/Processes.xsd'),
    ('omso', 'http://inspire.ec.europa.eu/schemas/omso/3.0', 'http://inspire.ec.europa.eu/schemas/omso/3.0/SpecialisedObservations.xsd'),
    ('pd', 'http://inspire.ec.europa.eu/schemas/pd/4.0', 'http://inspire.ec.europa.eu/schemas/pd/4.0/PopulationDistributionDemography.xsd'),
    ('pf', 'http://inspire.ec.europa.eu/schemas/pf/4.0', 'http://inspire.ec.europa.eu/schemas/pf/4.0/ProductionAndIndustrialFacilities.xsd'),
    ('plu', 'http://inspire.ec.europa.eu/schemas/plu/4.0', 'http://inspire.ec.europa.eu/schemas/plu/4.0/PlannedLandUse.xsd'),
    ('ps', 'http://inspire.ec.europa.eu/schemas/ps/4.0', 'http://inspire.ec.europa.eu/schemas/ps/4.0/ProtectedSites.xsd'),
    ('sd', 'http://inspire.ec.europa.eu/schemas/sd/4.0', 'http://inspire.ec.europa.eu/schemas/sd/4.0/SpeciesDistribution.xsd'),
    ('selu', 'http://inspire.ec.europa.eu/schemas/selu/4.0', 'http://inspire.ec.europa.eu/schemas/selu/4.0/SampledExistingLandUse.xsd'),
    ('so', 'http://inspire.ec.europa.eu/schemas/so/4.0', 'http://inspire.ec.europa.eu/schemas/so/4.0/Soil.xsd'),
    ('sr', 'http://inspire.ec.europa.eu/schemas/sr/4.0', 'http://inspire.ec.europa.eu/schemas/sr/4.0/SeaRegions.xsd'),
    ('su-core', 'http://inspire.ec.europa.eu/schemas/su-core/4.0', 'http://inspire.ec.europa.eu/schemas/su-core/4.0/StatisticalUnitCore.xsd'),
    ('su-grid', 'http://inspire.ec.europa.eu/schemas/su-grid/4.0', 'http://inspire.ec.europa.eu/schemas/su-grid/4.0/StatisticalUnitGrid.xsd'),
    ('su-vector', 'http://inspire.ec.europa.eu/schemas/su-vector/4.0', 'http://inspire.ec.europa.eu/schemas/su-vector/4.0/StatisticalUnitVector.xsd'),
    ('su', 'http://inspire.ec.europa.eu/schemas/su/0.0', 'http://inspire.ec.europa.eu/schemas/su/0.0/StatisticalUnits.xsd'),
    ('tn-a', 'http://inspire.ec.europa.eu/schemas/tn-a/4.0', 'http://inspire.ec.europa.eu/schemas/tn-a/4.0/AirTransportNetwork.xsd'),
    ('tn-c', 'http://inspire.ec.europa.eu/schemas/tn-c/4.0', 'http://inspire.ec.europa.eu/schemas/tn-c/4.0/CableTransportNetwork.xsd'),
    ('tn-ra', 'http://inspire.ec.europa.eu/schemas/tn-ra/4.0', 'http://inspire.ec.europa.eu/schemas/tn-ra/4.0/RailwayTransportNetwork.xsd'),
    ('tn-ro', 'http://inspire.ec.europa.eu/schemas/tn-ro/4.0', 'http://inspire.ec.europa.eu/schemas/tn-ro/4.0/RoadTransportNetwork.xsd'),
    ('tn-w', 'http://inspire.ec.europa.eu/schemas/tn-w/4.0', 'http://inspire.ec.europa.eu/schemas/tn-w/4.0/WaterTransportNetwork.xsd'),
    ('tn', 'http://inspire.ec.europa.eu/schemas/tn/4.0', 'http://inspire.ec.europa.eu/schemas/tn/4.0/CommonTransportElements.xsd'),
    ('ugs', 'http://inspire.ec.europa.eu/schemas/ugs/0.0', 'http://inspire.ec.europa.eu/schemas/ugs/0.0/UtilityAndGovernmentalServices.xsd'),
    ('us-emf', 'http://inspire.ec.europa.eu/schemas/us-emf/4.0', 'http://inspire.ec.europa.eu/schemas/us-emf/4.0/EnvironmentalManagementFacilities.xsd'),
    ('us-govserv', 'http://inspire.ec.europa.eu/schemas/us-govserv/4.0', 'http://inspire.ec.europa.eu/schemas/us-govserv/4.0/GovernmentalServices.xsd'),
    ('us-net-common', 'http://inspire.ec.europa.eu/schemas/us-net-common/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-common/4.0/UtilityNetworksCommon.xsd'),
    ('us-net-el', 'http://inspire.ec.europa.eu/schemas/us-net-el/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-el/4.0/ElectricityNetwork.xsd'),
    ('us-net-ogc', 'http://inspire.ec.europa.eu/schemas/us-net-ogc/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-ogc/4.0/OilGasChemicalsNetwork.xsd'),
    ('us-net-sw', 'http://inspire.ec.europa.eu/schemas/us-net-sw/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-sw/4.0/SewerNetwork.xsd'),
    ('us-net-tc', 'http://inspire.ec.europa.eu/schemas/us-net-tc/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-tc/4.0/TelecommunicationsNetwork.xsd'),
    ('us-net-th', 'http://inspire.ec.europa.eu/schemas/us-net-th/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-th/4.0/ThermalNetwork.xsd'),
    ('us-net-wa', 'http://inspire.ec.europa.eu/schemas/us-net-wa/4.0', 'http://inspire.ec.europa.eu/schemas/us-net-wa/4.0/WaterNetwork.xsd'),
    ('wfd', 'http://inspire.ec.europa.eu/schemas/wfd/0.0', 'http://inspire.ec.europa.eu/schemas/wfd/0.0/WaterFrameworkDirective.xsd'),
]

##

_XMLNS = 'xmlns'
_XSI = 'xsi'
_XSI_URL = 'http://www.w3.org/2001/XMLSchema-instance'


class NamespaceManager:
    def __init__(self):
        self._pfx_to_uri = {}
        self._uri_to_pfx = {}
        self._schema_loc = {}
        self._adhoc_ns_count = 0

        for pfx, uri, sl in _KNOWN_NAMESPACES:
            self.add(pfx, uri, sl)

    def add(self, pfx, uri, schema_loc=''):
        self._pfx_to_uri[pfx] = uri
        self._uri_to_pfx[uri] = pfx
        if schema_loc:
            self._schema_loc[uri] = self._schema_loc[pfx] = schema_loc

    def prefix(self, uri, generate_missing=False):
        pfx = self._uri_to_pfx.get(uri)
        if pfx or not generate_missing:
            return pfx
        self._adhoc_ns_count += 1
        pfx = 'ns' + str(self._adhoc_ns_count)
        self.add(pfx, uri)
        return pfx

    def uri(self, pfx):
        return self._pfx_to_uri.get(pfx)

    def schema_location(self, uri_or_pfx):
        return self._schema_loc.get(uri_or_pfx)

    def declarations(self, default_prefix=None, for_element=None, prefixes=None, with_schema_locations=True):
        pfx_set = set()

        if for_element:
            self._collect_prefixes(for_element, pfx_set)
        if prefixes:
            pfx_set.update(prefixes)
        if default_prefix:
            pfx_set.add(default_prefix)

        atts = []
        schema_locations = []

        for pfx in pfx_set:
            uri = self._pfx_to_uri.get(pfx)
            if not uri and pfx in self._uri_to_pfx:
                # ns URI given instead of a prefix?
                uri = pfx
            if not uri:
                raise error.WriteError(f'unknown namespace {pfx!r}')
            atts.append((_XMLNS if pfx == default_prefix else _XMLNS + ':' + pfx, uri))
            if with_schema_locations:
                sch = self._schema_loc.get(uri)
                if sch:
                    schema_locations.append(uri)
                    schema_locations.append(sch)

        if schema_locations:
            atts.append((_XMLNS + ':' + _XSI, _XSI_URL))
            atts.append((_XSI + ':schemaLocation', ' '.join(schema_locations)))

        return dict(sorted(atts))

    def _collect_prefixes(self, el, prefixes):
        if ':' in el.tag:
            pfx, _, _ = el.tag.partition(':')
            prefixes.add(pfx)

        for key in el.attributes:
            if ':' in key:
                pfx, _, name = el.tag.partition(':')
                prefixes.add(name if pfx == _XMLNS else pfx)

        for ch in el:
            self._collect_prefixes(ch, prefixes)


_namespaces = NamespaceManager()
