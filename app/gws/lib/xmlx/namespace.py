"""XML namespace manager.

Manage XML namespaces.
"""

import re

import gws
import gws.types as t

from . import error


class NamespaceManager:
    def __init__(self):
        self._prefix_to_uri = {}
        self._uri_to_prefix = {}
        self._prefix_to_version = {}
        self._prefix_to_schema = {}

        for ln in _KNOWN_NAMESPACES.strip().split('\n'):
            ln = ln.strip()
            if not ln:
                continue
            pfx, version, uri, schema = [s.strip() for s in ln.split('|')]
            if uri:
                uri = 'http://' + uri
            self._prefix_to_uri[pfx] = uri
            self._uri_to_prefix[uri] = pfx
            if version:
                self._prefix_to_version[pfx] = version
                self._uri_to_prefix[uri + '/' + version] = pfx
            if schema:
                self._prefix_to_schema[pfx] = 'http://' + schema

    def uri_for_prefix(self, prefix: str) -> str | None:
        """Returns an Uri for a canonical prefix.

        Args:
            prefix: Canonical prefix, e.g. ``gml``.
        Returns:
            Namespace Uri, e.g. ``http://www.opengis.net/gml``.
        """
        return self._prefix_to_uri.get(prefix)

    def prefix_for_uri(self, uri):
        """Returns the canonical prefix for a namespace Uri.

        If the Uri ends with a version number, and there is no prefix
        for this specific Uri, the function tries to locate a non-versioned Uri.
        That is, ``http://www.opengis.net/gml/3.2 -> http://www.opengis.net/gml -> 'gml'``.

        Args:
            uri: An Uri like ``http://www.opengis.net/gml``.
        Returns:
            Canonical prefix, e.g. ``gml``.
        """

        pfx = self._uri_to_prefix.get(uri)
        if pfx:
            return pfx

        # try with a version removed
        m = re.match(r'(.+?)/[\d.]+$', uri)
        if m:
            pfx = self._uri_to_prefix.get(m.group(1))
            if pfx:
                # cache for future use
                self._uri_to_prefix[uri] = pfx
                return pfx

    def parse_name(self, name):
        """Parses an XML name.

        Args:
            name: XML name.
        Returns:
            A triple ``(canonical prefix, Uri, proper name)``.
        """

        if name and name[0] == '{':
            s = name[1:].split('}')
            return self.prefix_for_uri(s[0]), s[0], s[1]

        if ':' in name:
            s = name.split(':')
            return s[0], self.uri_for_prefix(s[0]), s[1]

        return '', '', name

    def unqualify(self, name):
        """Returns an unqualified XML name."""

        _, _, name = self.parse_name(name)
        return name

    def qualify(self, name: str, default_prefix: str = ''):
        """Qualifies an XML name.

        If the name does not contain a namespace, returns it as is.
        If the namespace prefix is equal to the default prefix, returns a proper name.
        Othewise, returns ``prefix:name``.

        Args:
            name: An XML name.
            default_prefix: Default namespace prefix.
        Returns:
            A quailified name.
        """

        pfx, uri, name = self.parse_name(name)
        if not pfx:
            if not uri:
                return name
            raise error.NamespaceError(f'unknown namespace uri {uri!r}')
        if pfx == default_prefix:
            return name
        return pfx + ':' + name

    def clarkify(self, name):
        """Returns an XML name in the Clark notation."""

        pfx, uri, name = self.parse_name(name)
        if not uri:
            if not pfx:
                return name
            raise error.NamespaceError(f'unknown namespace prefix {pfx!r}')
        return '{' + uri + '}' + name

    def register(self, prefix: str, uri: str, version: str = '', schema: str = ''):
        """Registers a new namespace.

        Args:
            prefix: Canonical prefix.
            uri: Namespace Uri.
            version: Namespace version to use in ``xmlns`` declarations.
            schema: Schema Uri.
        """

        self._prefix_to_uri[prefix] = uri
        self._uri_to_prefix[uri] = prefix
        if version is not None:
            self._prefix_to_version[prefix] = version
        if schema is not None:
            self._prefix_to_schema[prefix] = schema

    def declarations(
            self,
            default_prefix: str = None,
            for_element: gws.IXmlElement = None,
            extra_prefixes: t.List[str] = None,
            with_schema_locations: bool = False,
    ) -> dict:
        """Returns an xmlns declaration block as dictionary of attributes.

        Args:
            default_prefix: Default namespace prefix. For example,
                if ``default_prefix`` is ``gml``, the declaration will be ``xmls=http://www.opengis.net/gml``).
            for_element: If given, collect namespaces from this element
                and its descendants.
            extra_prefixes: Extra prefixes to create declarations for.
            with_schema_locations: Add the "schema location" attribute.

        Returns:
            A dict of attributes.
        """

        pfx_set = set()

        if for_element:
            self._collect_prefixes(for_element, pfx_set)
        if extra_prefixes:
            pfx_set.update(extra_prefixes)
        if default_prefix:
            pfx_set.add(default_prefix)

        atts = []
        schema_locations = []

        for pfx in pfx_set:
            uri = self._prefix_to_uri.get(pfx)
            if not uri and pfx in self._uri_to_prefix:
                # ns URI given instead of a prefix?
                uri = pfx
            if not uri:
                raise error.NamespaceError(f'unknown namespace {pfx!r}')
            version = self._prefix_to_version.get(pfx)
            if version:
                uri += '/' + version
            atts.append((_XMLNS if pfx == default_prefix else _XMLNS + ':' + pfx, uri))
            if with_schema_locations:
                sch = self._prefix_to_schema.get(pfx)
                if sch:
                    schema_locations.append(uri)
                    schema_locations.append(sch)

        if schema_locations:
            atts.append((_XMLNS + ':' + _XSI, _XSI_URL))
            atts.append((_XSI + ':schemaLocation', ' '.join(schema_locations)))

        return dict(sorted(atts))

    def _collect_prefixes(self, el: gws.IXmlElement, pfx_set):
        pfx, uri, name = self.parse_name(el.tag)
        if pfx:
            pfx_set.add(pfx)

        for key in el.attrib:
            pfx, uri, name = self.parse_name(key)
            if pfx:
                pfx_set.add(name if pfx == _XMLNS else pfx)

        for sub in el:
            self._collect_prefixes(sub, pfx_set)


# prefix | preferred version for output | uri | schema location

_KNOWN_NAMESPACES = """
xml              |       | www.w3.org/XML/1998/namespace                 |
html             |       | www.w3.org/1999/xhtml                         |
wsdl             |       | schemas.xmlsoap.org/wsdl                      |
xs               |       | www.w3.org/2001/XMLSchema                     |
xsd              |       | www.w3.org/2001/XMLSchema                     |
xsi              |       | www.w3.org/2001/XMLSchema-instance            |
xlink            |       | www.w3.org/1999/xlink                         | https://www.w3.org/XML/2008/06/xlink.xsd
rdf              |       | www.w3.org/1999/02/22-rdf-syntax-ns           |
soap             |       | www.w3.org/2003/05/soap-envelope              | https://www.w3.org/2003/05/soap-envelope/

csw              | 2.0.2 | www.opengis.net/cat/csw                       | schemas.opengis.net/csw/2.0.2/csw.xsd
fes              | 2.0   | www.opengis.net/fes                           | schemas.opengis.net/filter/2.0/filterAll.xsd
gml              | 3.2   | www.opengis.net/gml                           | schemas.opengis.net/gml/3.2.1/gml.xsd
gmlcov           | 1.0   | www.opengis.net/gmlcov                        | schemas.opengis.net/gmlcov/1.0/gmlcovAll.xsd
ogc              |       | www.opengis.net/ogc                           | schemas.opengis.net/filter/1.1.0/filter.xsd
ows              | 1.1   | www.opengis.net/ows                           | schemas.opengis.net/ows/1.0.0/owsAll.xsd
sld              |       | www.opengis.net/sld                           | schemas.opengis.net/sld/1.1/sldAll.xsd
swe              | 2.0   | www.opengis.net/swe                           | schemas.opengis.net/sweCommon/2.0/swe.xsd
wcs              | 2.0   | www.opengis.net/wcs                           | schemas.opengis.net/wcs/1.0.0/wcsAll.xsd
wcscrs           | 1.0   | www.opengis.net/wcs/crs                       |
wcsint           | 1.0   | www.opengis.net/wcs/interpolation             |
wcsscal          | 1.0   | www.opengis.net/wcs/scaling                   |
wfs              | 2.0   | www.opengis.net/wfs                           | schemas.opengis.net/wfs/2.0/wfs.xsd
wms              |       | www.opengis.net/wms                           | schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd
wmts             | 1.0   | www.opengis.net/wmts                          | schemas.opengis.net/wmts/1.0/wmts.xsd

dc               | 1.1   | purl.org/dc/elements                          | schemas.opengis.net/csw/2.0.2/rec-dcmes.xsd
dcm              |       | purl.org/dc/dcmitype                          | dublincore.org/schemas/xmls/qdc/2008/02/11/dcmitype.xsd
dct              |       | purl.org/dc/terms                             | schemas.opengis.net/csw/2.0.2/rec-dcterms.xsd

gco              |       | www.isotc211.org/2005/gco                     | schemas.opengis.net/iso/19139/20070417/gco/gco.xsd
gmd              |       | www.isotc211.org/2005/gmd                     | schemas.opengis.net/csw/2.0.2/profiles/apiso/1.0.0/apiso.xsd
gmx              |       | www.isotc211.org/2005/gmx                     | schemas.opengis.net/iso/19139/20070417/gmx/gmx.xsd
srv              |       | www.isotc211.org/2005/srv                     | schemas.opengis.net/iso/19139/20070417/srv/1.0/srv.xsd

inspire_dls      | 1.0   | inspire.ec.europa.eu/schemas/inspire_dls      | inspire.ec.europa.eu/schemas/inspire_dls/1.0/inspire_dls.xsd
inspire_ds       | 1.0   | inspire.ec.europa.eu/schemas/inspire_ds       | inspire.ec.europa.eu/schemas/inspire_ds/1.0/inspire_ds.xsd
inspire_vs       | 1.0   | inspire.ec.europa.eu/schemas/inspire_vs       | inspire.ec.europa.eu/schemas/inspire_vs/1.0/inspire_vs.xsd
inspire_vs_ows11 | 1.0   | inspire.ec.europa.eu/schemas/inspire_vs_ows11 | inspire.ec.europa.eu/schemas/inspire_vs_ows11/1.0/inspire_vs_ows_11.xsd
inspire_common   | 1.0   | inspire.ec.europa.eu/schemas/common           | inspire.ec.europa.eu/schemas/common/1.0/common.xsd

ac-mf            | 4.0   | inspire.ec.europa.eu/schemas/ac-mf            | inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd
ac               | 4.0   | inspire.ec.europa.eu/schemas/ac-mf            | inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd
mf               | 4.0   | inspire.ec.europa.eu/schemas/ac-mf            | inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd
act-core         | 4.0   | inspire.ec.europa.eu/schemas/act-core         | inspire.ec.europa.eu/schemas/act-core/4.0/ActivityComplex_Core.xsd
ad               | 4.0   | inspire.ec.europa.eu/schemas/ad               | inspire.ec.europa.eu/schemas/ad/4.0/Addresses.xsd
af               | 4.0   | inspire.ec.europa.eu/schemas/af               | inspire.ec.europa.eu/schemas/af/4.0/AgriculturalAndAquacultureFacilities.xsd
am               | 4.0   | inspire.ec.europa.eu/schemas/am               | inspire.ec.europa.eu/schemas/am/4.0/AreaManagementRestrictionRegulationZone.xsd
au               | 4.0   | inspire.ec.europa.eu/schemas/au               | inspire.ec.europa.eu/schemas/au/4.0/AdministrativeUnits.xsd
base             | 3.3   | inspire.ec.europa.eu/schemas/base             | inspire.ec.europa.eu/schemas/base/3.3/BaseTypes.xsd
base2            | 2.0   | inspire.ec.europa.eu/schemas/base2            | inspire.ec.europa.eu/schemas/base2/2.0/BaseTypes2.xsd
br               | 4.0   | inspire.ec.europa.eu/schemas/br               | inspire.ec.europa.eu/schemas/br/4.0/Bio-geographicalRegions.xsd
bu-base          | 4.0   | inspire.ec.europa.eu/schemas/bu-base          | inspire.ec.europa.eu/schemas/bu-base/4.0/BuildingsBase.xsd
bu-core2d        | 4.0   | inspire.ec.europa.eu/schemas/bu-core2d        | inspire.ec.europa.eu/schemas/bu-core2d/4.0/BuildingsCore2D.xsd
bu-core3d        | 4.0   | inspire.ec.europa.eu/schemas/bu-core3d        | inspire.ec.europa.eu/schemas/bu-core3d/4.0/BuildingsCore3D.xsd
bu               | 0.0   | inspire.ec.europa.eu/schemas/bu               | inspire.ec.europa.eu/schemas/bu/0.0/Buildings.xsd
cp               | 4.0   | inspire.ec.europa.eu/schemas/cp               | inspire.ec.europa.eu/schemas/cp/4.0/CadastralParcels.xsd
cvbase           | 2.0   | inspire.ec.europa.eu/schemas/cvbase           | inspire.ec.europa.eu/schemas/cvbase/2.0/CoverageBase.xsd
cvgvp            | 0.1   | inspire.ec.europa.eu/schemas/cvgvp            | inspire.ec.europa.eu/schemas/cvgvp/0.1/CoverageGVP.xsd
ef               | 4.0   | inspire.ec.europa.eu/schemas/ef               | inspire.ec.europa.eu/schemas/ef/4.0/EnvironmentalMonitoringFacilities.xsd
el-bas           | 4.0   | inspire.ec.europa.eu/schemas/el-bas           | inspire.ec.europa.eu/schemas/el-bas/4.0/ElevationBaseTypes.xsd
el-cov           | 4.0   | inspire.ec.europa.eu/schemas/el-cov           | inspire.ec.europa.eu/schemas/el-cov/4.0/ElevationGridCoverage.xsd
el-tin           | 4.0   | inspire.ec.europa.eu/schemas/el-tin           | inspire.ec.europa.eu/schemas/el-tin/4.0/ElevationTin.xsd
el-vec           | 4.0   | inspire.ec.europa.eu/schemas/el-vec           | inspire.ec.europa.eu/schemas/el-vec/4.0/ElevationVectorElements.xsd
elu              | 4.0   | inspire.ec.europa.eu/schemas/elu              | inspire.ec.europa.eu/schemas/elu/4.0/ExistingLandUse.xsd
er-b             | 4.0   | inspire.ec.europa.eu/schemas/er-b             | inspire.ec.europa.eu/schemas/er-b/4.0/EnergyResourcesBase.xsd
er-c             | 4.0   | inspire.ec.europa.eu/schemas/er-c             | inspire.ec.europa.eu/schemas/er-c/4.0/EnergyResourcesCoverage.xsd
er-v             | 4.0   | inspire.ec.europa.eu/schemas/er-v             | inspire.ec.europa.eu/schemas/er-v/4.0/EnergyResourcesVector.xsd
er               | 0.0   | inspire.ec.europa.eu/schemas/er               | inspire.ec.europa.eu/schemas/er/0.0/EnergyResources.xsd
gaz              | 3.2   | inspire.ec.europa.eu/schemas/gaz              | inspire.ec.europa.eu/schemas/gaz/3.2/Gazetteer.xsd
ge-core          | 4.0   | inspire.ec.europa.eu/schemas/ge-core          | inspire.ec.europa.eu/schemas/ge-core/4.0/GeologyCore.xsd
ge               | 0.0   | inspire.ec.europa.eu/schemas/ge               | inspire.ec.europa.eu/schemas/ge/0.0/Geology.xsd
ge_gp            | 4.0   | inspire.ec.europa.eu/schemas/ge_gp            | inspire.ec.europa.eu/schemas/ge_gp/4.0/GeophysicsCore.xsd
ge_hg            | 4.0   | inspire.ec.europa.eu/schemas/ge_hg            | inspire.ec.europa.eu/schemas/ge_hg/4.0/HydrogeologyCore.xsd
gelu             | 4.0   | inspire.ec.europa.eu/schemas/gelu             | inspire.ec.europa.eu/schemas/gelu/4.0/GriddedExistingLandUse.xsd
geoportal        | 1.0   | inspire.ec.europa.eu/schemas/geoportal        | inspire.ec.europa.eu/schemas/geoportal/1.0/geoportal.xsd
gn               | 4.0   | inspire.ec.europa.eu/schemas/gn               | inspire.ec.europa.eu/schemas/gn/4.0/GeographicalNames.xsd
hb               | 4.0   | inspire.ec.europa.eu/schemas/hb               | inspire.ec.europa.eu/schemas/hb/4.0/HabitatsAndBiotopes.xsd
hh               | 4.0   | inspire.ec.europa.eu/schemas/hh               | inspire.ec.europa.eu/schemas/hh/4.0/HumanHealth.xsd
hy-n             | 4.0   | inspire.ec.europa.eu/schemas/hy-n             | inspire.ec.europa.eu/schemas/hy-n/4.0/HydroNetwork.xsd
hy-p             | 4.0   | inspire.ec.europa.eu/schemas/hy-p             | inspire.ec.europa.eu/schemas/hy-p/4.0/HydroPhysicalWaters.xsd
hy               | 4.0   | inspire.ec.europa.eu/schemas/hy               | inspire.ec.europa.eu/schemas/hy/4.0/HydroBase.xsd
lc               | 0.0   | inspire.ec.europa.eu/schemas/lc               | inspire.ec.europa.eu/schemas/lc/0.0/LandCover.xsd
lcn              | 4.0   | inspire.ec.europa.eu/schemas/lcn              | inspire.ec.europa.eu/schemas/lcn/4.0/LandCoverNomenclature.xsd
lcr              | 4.0   | inspire.ec.europa.eu/schemas/lcr              | inspire.ec.europa.eu/schemas/lcr/4.0/LandCoverRaster.xsd
lcv              | 4.0   | inspire.ec.europa.eu/schemas/lcv              | inspire.ec.europa.eu/schemas/lcv/4.0/LandCoverVector.xsd
lu               | 4.0   | inspire.ec.europa.eu/schemas/lunom            | inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd
lunom            | 4.0   | inspire.ec.europa.eu/schemas/lunom            | inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd
mr-core          | 4.0   | inspire.ec.europa.eu/schemas/mr-core          | inspire.ec.europa.eu/schemas/mr-core/4.0/MineralResourcesCore.xsd
mu               |       | inspire.ec.europa.eu/schemas/mu/3.0rc3        | inspire.ec.europa.eu/schemas/mu/3.0rc3/MaritimeUnits.xsd
net              | 4.0   | inspire.ec.europa.eu/schemas/net              | inspire.ec.europa.eu/schemas/net/4.0/Network.xsd
nz-core          | 4.0   | inspire.ec.europa.eu/schemas/nz-core          | inspire.ec.europa.eu/schemas/nz-core/4.0/NaturalRiskZonesCore.xsd
nz               | 0.0   | inspire.ec.europa.eu/schemas/nz               | inspire.ec.europa.eu/schemas/nz/0.0/NaturalRiskZones.xsd
of               | 4.0   | inspire.ec.europa.eu/schemas/of               | inspire.ec.europa.eu/schemas/of/4.0/OceanFeatures.xsd
oi               | 4.0   | inspire.ec.europa.eu/schemas/oi               | inspire.ec.europa.eu/schemas/oi/4.0/Orthoimagery.xsd
omop             | 3.0   | inspire.ec.europa.eu/schemas/omop             | inspire.ec.europa.eu/schemas/omop/3.0/ObservableProperties.xsd
omor             | 3.0   | inspire.ec.europa.eu/schemas/omor             | inspire.ec.europa.eu/schemas/omor/3.0/ObservationReferences.xsd
ompr             | 3.0   | inspire.ec.europa.eu/schemas/ompr             | inspire.ec.europa.eu/schemas/ompr/3.0/Processes.xsd
omso             | 3.0   | inspire.ec.europa.eu/schemas/omso             | inspire.ec.europa.eu/schemas/omso/3.0/SpecialisedObservations.xsd
pd               | 4.0   | inspire.ec.europa.eu/schemas/pd               | inspire.ec.europa.eu/schemas/pd/4.0/PopulationDistributionDemography.xsd
pf               | 4.0   | inspire.ec.europa.eu/schemas/pf               | inspire.ec.europa.eu/schemas/pf/4.0/ProductionAndIndustrialFacilities.xsd
plu              | 4.0   | inspire.ec.europa.eu/schemas/plu              | inspire.ec.europa.eu/schemas/plu/4.0/PlannedLandUse.xsd
ps               | 4.0   | inspire.ec.europa.eu/schemas/ps               | inspire.ec.europa.eu/schemas/ps/4.0/ProtectedSites.xsd
sd               | 4.0   | inspire.ec.europa.eu/schemas/sd               | inspire.ec.europa.eu/schemas/sd/4.0/SpeciesDistribution.xsd
selu             | 4.0   | inspire.ec.europa.eu/schemas/selu             | inspire.ec.europa.eu/schemas/selu/4.0/SampledExistingLandUse.xsd
so               | 4.0   | inspire.ec.europa.eu/schemas/so               | inspire.ec.europa.eu/schemas/so/4.0/Soil.xsd
sr               | 4.0   | inspire.ec.europa.eu/schemas/sr               | inspire.ec.europa.eu/schemas/sr/4.0/SeaRegions.xsd
su-core          | 4.0   | inspire.ec.europa.eu/schemas/su-core          | inspire.ec.europa.eu/schemas/su-core/4.0/StatisticalUnitCore.xsd
su-grid          | 4.0   | inspire.ec.europa.eu/schemas/su-grid          | inspire.ec.europa.eu/schemas/su-grid/4.0/StatisticalUnitGrid.xsd
su-vector        | 4.0   | inspire.ec.europa.eu/schemas/su-vector        | inspire.ec.europa.eu/schemas/su-vector/4.0/StatisticalUnitVector.xsd
su               | 0.0   | inspire.ec.europa.eu/schemas/su               | inspire.ec.europa.eu/schemas/su/0.0/StatisticalUnits.xsd
tn-a             | 4.0   | inspire.ec.europa.eu/schemas/tn-a             | inspire.ec.europa.eu/schemas/tn-a/4.0/AirTransportNetwork.xsd
tn-c             | 4.0   | inspire.ec.europa.eu/schemas/tn-c             | inspire.ec.europa.eu/schemas/tn-c/4.0/CableTransportNetwork.xsd
tn-ra            | 4.0   | inspire.ec.europa.eu/schemas/tn-ra            | inspire.ec.europa.eu/schemas/tn-ra/4.0/RailwayTransportNetwork.xsd
tn-ro            | 4.0   | inspire.ec.europa.eu/schemas/tn-ro            | inspire.ec.europa.eu/schemas/tn-ro/4.0/RoadTransportNetwork.xsd
tn-w             | 4.0   | inspire.ec.europa.eu/schemas/tn-w             | inspire.ec.europa.eu/schemas/tn-w/4.0/WaterTransportNetwork.xsd
tn               | 4.0   | inspire.ec.europa.eu/schemas/tn               | inspire.ec.europa.eu/schemas/tn/4.0/CommonTransportElements.xsd
ugs              | 0.0   | inspire.ec.europa.eu/schemas/ugs              | inspire.ec.europa.eu/schemas/ugs/0.0/UtilityAndGovernmentalServices.xsd
us-emf           | 4.0   | inspire.ec.europa.eu/schemas/us-emf           | inspire.ec.europa.eu/schemas/us-emf/4.0/EnvironmentalManagementFacilities.xsd
us-govserv       | 4.0   | inspire.ec.europa.eu/schemas/us-govserv       | inspire.ec.europa.eu/schemas/us-govserv/4.0/GovernmentalServices.xsd
us-net-common    | 4.0   | inspire.ec.europa.eu/schemas/us-net-common    | inspire.ec.europa.eu/schemas/us-net-common/4.0/UtilityNetworksCommon.xsd
us-net-el        | 4.0   | inspire.ec.europa.eu/schemas/us-net-el        | inspire.ec.europa.eu/schemas/us-net-el/4.0/ElectricityNetwork.xsd
us-net-ogc       | 4.0   | inspire.ec.europa.eu/schemas/us-net-ogc       | inspire.ec.europa.eu/schemas/us-net-ogc/4.0/OilGasChemicalsNetwork.xsd
us-net-sw        | 4.0   | inspire.ec.europa.eu/schemas/us-net-sw        | inspire.ec.europa.eu/schemas/us-net-sw/4.0/SewerNetwork.xsd
us-net-tc        | 4.0   | inspire.ec.europa.eu/schemas/us-net-tc        | inspire.ec.europa.eu/schemas/us-net-tc/4.0/TelecommunicationsNetwork.xsd
us-net-th        | 4.0   | inspire.ec.europa.eu/schemas/us-net-th        | inspire.ec.europa.eu/schemas/us-net-th/4.0/ThermalNetwork.xsd
us-net-wa        | 4.0   | inspire.ec.europa.eu/schemas/us-net-wa        | inspire.ec.europa.eu/schemas/us-net-wa/4.0/WaterNetwork.xsd
wfd              | 0.0   | inspire.ec.europa.eu/schemas/wfd              | inspire.ec.europa.eu/schemas/wfd/0.0/WaterFrameworkDirective.xsd
"""

##

_XMLNS = 'xmlns'
_XSI = 'xsi'
_XSI_URL = 'http://www.w3.org/2001/XMLSchema-instance'

_mgr = NamespaceManager()

uri_for_prefix = _mgr.uri_for_prefix
prefix_for_uri = _mgr.prefix_for_uri
parse_name = _mgr.parse_name
unqualify = _mgr.unqualify
qualify = _mgr.qualify
clarkify = _mgr.clarkify
register = _mgr.register
declarations = _mgr.declarations
