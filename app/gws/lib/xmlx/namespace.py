"""XML namespace manager.

Maintains a registry of XML namespaces (well-known and custom).
"""

from typing import Optional
import re

import gws
from . import error

_ALL: list[gws.XmlNamespace] = []


class _Index:
    uid = {}
    xmlns = {}
    uri = {}


_INDEX = _Index()


def register(ns: gws.XmlNamespace):
    """Register a Namespace.

    Args:
        ns: Namespace
    """

    if ns.uid not in _INDEX.uid:
        _ALL.append(ns)
        _build_index()


def _build_index():
    _INDEX.uid = {}
    _INDEX.xmlns = {}
    _INDEX.uri = {}

    for ns in _ALL:
        _INDEX.uid[ns.uid] = ns

        if ns.xmlns not in _INDEX.xmlns or ns.isDefault:
            _INDEX.xmlns[ns.xmlns] = ns

        _INDEX.uri[ns.uri] = ns
        if ns.version and not ns.uri.endswith('/' + ns.version):
            _INDEX.uri[ns.uri + '/' + ns.version] = ns


def get(uid: str) -> Optional[gws.XmlNamespace]:
    """Locate the Namespace by a uid."""

    return _INDEX.uid.get(uid)


def find_by_xmlns(xmlns: str) -> Optional[gws.XmlNamespace]:
    """Locate the Namespace by a prefix."""

    return _INDEX.xmlns.get(xmlns)


def find_by_uri(uri: str) -> Optional[gws.XmlNamespace]:
    """Locate the Namespace by an Uri."""

    return _INDEX.uri.get(uri)


def split_name(name: str) -> tuple[str, str, str]:
    """Parses an XML name in a xmlns: or Clark notation.

    Args:
        name: XML name.

    Returns:
        A tuple ``(xmlns-prefix, uri, proper name)``.
    """

    if not name:
        return '', '', name

    if name[0] == '{':
        s = name.split('}')
        return '', s[0][1:], s[1]

    if ':' in name:
        s = name.split(':')
        return s[0], '', s[1]

    return '', '', name


def extract(name: str) -> tuple[Optional[gws.XmlNamespace], str]:
    """Extract A Namespace object from a qualified name.

    Args:
        name: XML name.

    Returns:
        A tuple ``(XmlNamespace, proper name)``
    """

    xmlns, uri, pname = split_name(name)

    if xmlns:
        ns = find_by_xmlns(xmlns)
        if not ns:
            raise error.NamespaceError(f'unknown namespace {xmlns!r}')
        return ns, pname

    if uri:
        ns = find_by_uri(uri)
        if not ns:
            raise error.NamespaceError(f'unknown namespace uri {uri!r}')
        return ns, pname

    return None, pname


def qualify_name(name: str, ns: Optional[gws.XmlNamespace] = None, replace: bool = False) -> str:
    """Qualifies an XML name.

    If the name contains a namespace, return as is, otherwise, prepend the namespace.

    Args:
        name: An XML name.
        ns: A namespace.
        replace: If true, replace the existing namespace.

    Returns:
        A qualified name.
    """

    ns2, pname = extract(name)
    if ns2 and not replace:
        return ns2.xmlns + ':' + pname
    if ns:
        return ns.xmlns + ':' + pname
    return pname


def unqualify_name(name: str) -> str:
    """Returns an unqualified XML name."""

    _, _, name = split_name(name)
    return name


def unqualify_default(name: str, default_namespace: gws.XmlNamespace) -> str:
    """Removes the default namespace prefix.

    If the name contains the default namespace, remove it, otherwise return the name as is.

    Args:
        name: An XML name.
        default_namespace: A namespace.

    Returns:
        The name.
    """

    ns, pname = extract(name)
    if ns and ns.uid == default_namespace.uid:
        return pname
    if ns:
        return ns.xmlns + ':' + pname
    return name


def clarkify_name(name: str) -> str:
    """Returns an XML name in the Clark notation.

    Args:
        name: A qualified XML name.

    Returns:
        The XML name in Clark notation.
    """

    ns, pname = extract(name)
    if ns:
        return '{' + ns.uri + '}' + pname
    return pname


def declarations(
        for_element: gws.XmlElement = None,
        default_namespace: Optional[gws.XmlNamespace] = None,
        extra_namespaces: Optional[list[gws.XmlNamespace]] = None,
        xmlns_replacements: Optional[dict[str, str]] = None,
        with_schema_locations: bool = False,
) -> dict:
    """Returns an xmlns declaration block as dictionary of attributes.

    Args:
        default_namespace: Default namespace.
        for_element: If given, collect namespaces from this element and its descendants.
        extra_namespaces: Extra namespaces to create declarations for.
        xmlns_replacements: A mapping of namespace UIDs to custom namespace prefixes.
        with_schema_locations: Add the "schema location" attribute.

    Returns:
        A dict of attributes.
    """

    ns_map = {}

    if for_element:
        _collect_namespaces(for_element, ns_map)
    if extra_namespaces:
        for ns in extra_namespaces:
            ns_map[ns.uid] = ns
    if default_namespace:
        ns_map[default_namespace.uid] = default_namespace

    atts = []
    schemas = []

    for ns in ns_map.values():
        if default_namespace and ns.uid == default_namespace.uid:
            a = _XMLNS
        elif xmlns_replacements and ns.uid in xmlns_replacements:
            a = _XMLNS + ':' + xmlns_replacements[ns.uid]
        else:
            a = _XMLNS + ':' + ns.xmlns
        atts.append((a, ns.uri))

        if with_schema_locations and ns.schemaLocation:
            schemas.append(ns.uri)
            schemas.append(ns.schemaLocation)

    if schemas:
        atts.append((_XMLNS + ':' + _XSI, _XSI_URL))
        atts.append((_XSI + ':schemaLocation', ' '.join(schemas)))

    return dict(sorted(atts))


def _collect_namespaces(el: gws.XmlElement, ns_map):
    ns, _ = extract(el.tag)
    if ns:
        ns_map[ns.uid] = ns

    for key in el.attrib:
        ns, _ = extract(key)
        if ns and ns.xmlns != _XMLNS:
            ns_map[ns.uid] = ns

    for sub in el:
        _collect_namespaces(sub, ns_map)


def _parse_versioned_uri(uri: str) -> tuple[str, str]:
    m = re.match(r'(.+?)/([\d.]+)$', uri)
    if m:
        return m.group(1), m.group(2)
    return '', uri


_KNOWN_NAMESPACES = """
# uid            | xmlns | D | version | uri                                           | schemaLocation                                                                                               

html             |       |   |         | www.w3.org/1999/xhtml                         |                                                                                                              
rdf              |       |   |         | www.w3.org/1999/02/22-rdf-syntax-ns           |                                                                                                              
soap             |       |   |         | www.w3.org/2003/05/soap-envelope              | www.w3.org/2003/05/soap-envelope/                                                                    
soap11           |       |   |         | schemas.xmlsoap.org/soap/envelope/            |                                                                                                              
wsdl             |       |   |         | schemas.xmlsoap.org/wsdl                      |                                                                                                              
xlink            |       |   |         | www.w3.org/1999/xlink                         | www.w3.org/XML/2008/06/xlink.xsd                                                                     
xml              |       |   |         | www.w3.org/XML/1998/namespace                 |                                                                                                              
xs               |       |   |         | www.w3.org/2001/XMLSchema                     |                                                                                                              
xsd              |       |   |         | www.w3.org/2001/XMLSchema                     |                                                                                                              
xsi              |       |   |         | www.w3.org/2001/XMLSchema-instance            |                                                                                                              

csw              |       |   | 2.0.1   | www.opengis.net/cat/csw                       | schemas.opengis.net/csw/2.0.2/CSW-discovery.xsd                                                       
fes              |       |   | 2.0     | www.opengis.net/fes/2.0                       | schemas.opengis.net/filter/2.0/filterAll.xsd                                                          
gml              |       |   | 3.2     | www.opengis.net/gml/3.2                       | schemas.opengis.net/gml/3.2.1/gml.xsd                                                                 
gml21            | gml   | N | 2.1     | www.opengis.net/gml                           | schemas.opengis.net/gml/2.1.2/gml.xsd                                                                 
gmlcov           |       |   | 1.0     | www.opengis.net/gmlcov                        | schemas.opengis.net/gmlcov/1.0/gmlcovAll.xsd                                                          
ogc              |       |   |         | www.opengis.net/ogc                           | schemas.opengis.net/filter/1.1.0/filter.xsd                                                           
ows              |       |   | 2.0     | www.opengis.net/ows/2.0                       | schemas.opengis.net/ows/2.0/owsAll.xsd                                                                
ows11            | ows   | N | 1.1.0   | www.opengis.net/ows                           | schemas.opengis.net/ows/2.0/owsAll.xsd                                                                
sld              |       |   |         | www.opengis.net/sld                           | schemas.opengis.net/sld/1.1/sldAll.xsd                                                                
swe              |       |   | 2.0     | www.opengis.net/swe                           | schemas.opengis.net/sweCommon/2.0/swe.xsd                                                             
wcs              |       |   | 2.0     | www.opengis.net/wcs                           | schemas.opengis.net/wcs/1.0.0/wcsAll.xsd                                                              
wcscrs           |       |   | 1.0     | www.opengis.net/wcs/crs                       |                                                                                                              
wcsint           |       |   | 1.0     | www.opengis.net/wcs/interpolation             |                                                                                                              
wcsscal          |       |   | 1.0     | www.opengis.net/wcs/scaling                   |                                                                                                              
wfs              |       |   | 2.0     | www.opengis.net/wfs/2.0                       | schemas.opengis.net/wfs/2.0/wfs.xsd                                                                   
wfs11            | wfs   | N | 1.1.0   | www.opengis.net/wfs                           | schemas.opengis.net/wfs/1.1.0/wfs.xsd                                                                
wms              | wms   |   | 1.3.0   | www.opengis.net/wms                           | schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd                                                  
wms10            | wms   | N | 1.0.0   | www.opengis.net/wms                           | schemas.opengis.net/wms/1.0.0/capabilities_1_0_0.xml                                                  
wms11            | wms   | N | 1.1.0   | www.opengis.net/wms                           | schemas.opengis.net/wms/1.1.1/capabilities_1_1_1.xml                                                  
wmts             |       |   | 1.0     | www.opengis.net/wmts                          | schemas.opengis.net/wmts/1.0/wmts.xsd                                                                 

gco              |       |   |         | www.isotc211.org/2005/gco                     | schemas.opengis.net/iso/19139/20070417/gco/gco.xsd                                                    
gmd              |       |   |         | www.isotc211.org/2005/gmd                     | schemas.opengis.net/csw/2.0.2/profiles/apiso/1.0.0/apiso.xsd                                          
gmx              |       |   |         | www.isotc211.org/2005/gmx                     | schemas.opengis.net/iso/19139/20070417/gmx/gmx.xsd                                                    
srv              |       |   |         | www.isotc211.org/2005/srv                     | schemas.opengis.net/iso/19139/20070417/srv/1.0/srv.xsd                                                

dc               |       |   | 1.1     | purl.org/dc/elements                          | schemas.opengis.net/csw/2.0.2/rec-dcmes.xsd                                                           
dcm              |       |   |         | purl.org/dc/dcmitype                          | dublincore.org/schemas/xmls/qdc/2008/02/11/dcmitype.xsd                                               
dct              |       |   |         | purl.org/dc/terms                             | schemas.opengis.net/csw/2.0.2/rec-dcterms.xsd                                                         

ms               |       |   |         | mapserver.gis.umn.edu/mapserver               |                                                                                                              

inspire_dls      |       |   | 1.0     | inspire.ec.europa.eu/schemas/inspire_dls      | inspire.ec.europa.eu/schemas/inspire_dls/1.0/inspire_dls.xsd                                          
inspire_ds       |       |   | 1.0     | inspire.ec.europa.eu/schemas/inspire_ds       | inspire.ec.europa.eu/schemas/inspire_ds/1.0/inspire_ds.xsd                                            
inspire_vs       |       |   | 1.0     | inspire.ec.europa.eu/schemas/inspire_vs       | inspire.ec.europa.eu/schemas/inspire_vs/1.0/inspire_vs.xsd                                            
inspire_vs_ows11 |       |   | 1.0     | inspire.ec.europa.eu/schemas/inspire_vs_ows11 | inspire.ec.europa.eu/schemas/inspire_vs_ows11/1.0/inspire_vs_ows_11.xsd                               
inspire_common   |       |   | 1.0     | inspire.ec.europa.eu/schemas/common           | inspire.ec.europa.eu/schemas/common/1.0/common.xsd                                                    
ac-mf            |       |   | 4.0     | inspire.ec.europa.eu/schemas/ac-mf            | inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd 
ac               |       |   | 4.0     | inspire.ec.europa.eu/schemas/ac-mf            | inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd 
mf               |       |   | 4.0     | inspire.ec.europa.eu/schemas/ac-mf            | inspire.ec.europa.eu/schemas/ac-mf/4.0/AtmosphericConditionsandMeteorologicalGeographicalFeatures.xsd 
act-core         |       |   | 4.0     | inspire.ec.europa.eu/schemas/act-core         | inspire.ec.europa.eu/schemas/act-core/4.0/ActivityComplex_Core.xsd                                    
ad               |       |   | 4.0     | inspire.ec.europa.eu/schemas/ad               | inspire.ec.europa.eu/schemas/ad/4.0/Addresses.xsd                                                     
af               |       |   | 4.0     | inspire.ec.europa.eu/schemas/af               | inspire.ec.europa.eu/schemas/af/4.0/AgriculturalAndAquacultureFacilities.xsd                          
am               |       |   | 4.0     | inspire.ec.europa.eu/schemas/am               | inspire.ec.europa.eu/schemas/am/4.0/AreaManagementRestrictionRegulationZone.xsd                       
au               |       |   | 4.0     | inspire.ec.europa.eu/schemas/au               | inspire.ec.europa.eu/schemas/au/4.0/AdministrativeUnits.xsd                                           
base             |       |   | 3.3     | inspire.ec.europa.eu/schemas/base             | inspire.ec.europa.eu/schemas/base/3.3/BaseTypes.xsd                                                   
base2            |       |   | 2.0     | inspire.ec.europa.eu/schemas/base2            | inspire.ec.europa.eu/schemas/base2/2.0/BaseTypes2.xsd                                                 
br               |       |   | 4.0     | inspire.ec.europa.eu/schemas/br               | inspire.ec.europa.eu/schemas/br/4.0/Bio-geographicalRegions.xsd                                       
bu-base          |       |   | 4.0     | inspire.ec.europa.eu/schemas/bu-base          | inspire.ec.europa.eu/schemas/bu-base/4.0/BuildingsBase.xsd                                            
bu-core2d        |       |   | 4.0     | inspire.ec.europa.eu/schemas/bu-core2d        | inspire.ec.europa.eu/schemas/bu-core2d/4.0/BuildingsCore2D.xsd                                        
bu-core3d        |       |   | 4.0     | inspire.ec.europa.eu/schemas/bu-core3d        | inspire.ec.europa.eu/schemas/bu-core3d/4.0/BuildingsCore3D.xsd                                        
bu               |       |   | 0.0     | inspire.ec.europa.eu/schemas/bu               | inspire.ec.europa.eu/schemas/bu/0.0/Buildings.xsd                                                     
cp               |       |   | 4.0     | inspire.ec.europa.eu/schemas/cp               | inspire.ec.europa.eu/schemas/cp/4.0/CadastralParcels.xsd                                              
cvbase           |       |   | 2.0     | inspire.ec.europa.eu/schemas/cvbase           | inspire.ec.europa.eu/schemas/cvbase/2.0/CoverageBase.xsd                                              
cvgvp            |       |   | 0.1     | inspire.ec.europa.eu/schemas/cvgvp            | inspire.ec.europa.eu/schemas/cvgvp/0.1/CoverageGVP.xsd                                                
ef               |       |   | 4.0     | inspire.ec.europa.eu/schemas/ef               | inspire.ec.europa.eu/schemas/ef/4.0/EnvironmentalMonitoringFacilities.xsd                             
el-bas           |       |   | 4.0     | inspire.ec.europa.eu/schemas/el-bas           | inspire.ec.europa.eu/schemas/el-bas/4.0/ElevationBaseTypes.xsd                                        
el-cov           |       |   | 4.0     | inspire.ec.europa.eu/schemas/el-cov           | inspire.ec.europa.eu/schemas/el-cov/4.0/ElevationGridCoverage.xsd                                     
el-tin           |       |   | 4.0     | inspire.ec.europa.eu/schemas/el-tin           | inspire.ec.europa.eu/schemas/el-tin/4.0/ElevationTin.xsd                                              
el-vec           |       |   | 4.0     | inspire.ec.europa.eu/schemas/el-vec           | inspire.ec.europa.eu/schemas/el-vec/4.0/ElevationVectorElements.xsd                                   
elu              |       |   | 4.0     | inspire.ec.europa.eu/schemas/elu              | inspire.ec.europa.eu/schemas/elu/4.0/ExistingLandUse.xsd                                              
er-b             |       |   | 4.0     | inspire.ec.europa.eu/schemas/er-b             | inspire.ec.europa.eu/schemas/er-b/4.0/EnergyResourcesBase.xsd                                         
er-c             |       |   | 4.0     | inspire.ec.europa.eu/schemas/er-c             | inspire.ec.europa.eu/schemas/er-c/4.0/EnergyResourcesCoverage.xsd                                     
er-v             |       |   | 4.0     | inspire.ec.europa.eu/schemas/er-v             | inspire.ec.europa.eu/schemas/er-v/4.0/EnergyResourcesVector.xsd                                       
er               |       |   | 0.0     | inspire.ec.europa.eu/schemas/er               | inspire.ec.europa.eu/schemas/er/0.0/EnergyResources.xsd                                               
gaz              |       |   | 3.2     | inspire.ec.europa.eu/schemas/gaz              | inspire.ec.europa.eu/schemas/gaz/3.2/Gazetteer.xsd                                                    
ge-core          |       |   | 4.0     | inspire.ec.europa.eu/schemas/ge-core          | inspire.ec.europa.eu/schemas/ge-core/4.0/GeologyCore.xsd                                              
ge               |       |   | 0.0     | inspire.ec.europa.eu/schemas/ge               | inspire.ec.europa.eu/schemas/ge/0.0/Geology.xsd                                                       
ge_gp            |       |   | 4.0     | inspire.ec.europa.eu/schemas/ge_gp            | inspire.ec.europa.eu/schemas/ge_gp/4.0/GeophysicsCore.xsd                                             
ge_hg            |       |   | 4.0     | inspire.ec.europa.eu/schemas/ge_hg            | inspire.ec.europa.eu/schemas/ge_hg/4.0/HydrogeologyCore.xsd                                           
gelu             |       |   | 4.0     | inspire.ec.europa.eu/schemas/gelu             | inspire.ec.europa.eu/schemas/gelu/4.0/GriddedExistingLandUse.xsd                                      
geoportal        |       |   | 1.0     | inspire.ec.europa.eu/schemas/geoportal        | inspire.ec.europa.eu/schemas/geoportal/1.0/geoportal.xsd                                              
gn               |       |   | 4.0     | inspire.ec.europa.eu/schemas/gn               | inspire.ec.europa.eu/schemas/gn/4.0/GeographicalNames.xsd                                             
hb               |       |   | 4.0     | inspire.ec.europa.eu/schemas/hb               | inspire.ec.europa.eu/schemas/hb/4.0/HabitatsAndBiotopes.xsd                                           
hh               |       |   | 4.0     | inspire.ec.europa.eu/schemas/hh               | inspire.ec.europa.eu/schemas/hh/4.0/HumanHealth.xsd                                                   
hy-n             |       |   | 4.0     | inspire.ec.europa.eu/schemas/hy-n             | inspire.ec.europa.eu/schemas/hy-n/4.0/HydroNetwork.xsd                                                
hy-p             |       |   | 4.0     | inspire.ec.europa.eu/schemas/hy-p             | inspire.ec.europa.eu/schemas/hy-p/4.0/HydroPhysicalWaters.xsd                                         
hy               |       |   | 4.0     | inspire.ec.europa.eu/schemas/hy               | inspire.ec.europa.eu/schemas/hy/4.0/HydroBase.xsd                                                     
lc               |       |   | 0.0     | inspire.ec.europa.eu/schemas/lc               | inspire.ec.europa.eu/schemas/lc/0.0/LandCover.xsd                                                     
lcn              |       |   | 4.0     | inspire.ec.europa.eu/schemas/lcn              | inspire.ec.europa.eu/schemas/lcn/4.0/LandCoverNomenclature.xsd                                        
lcr              |       |   | 4.0     | inspire.ec.europa.eu/schemas/lcr              | inspire.ec.europa.eu/schemas/lcr/4.0/LandCoverRaster.xsd                                              
lcv              |       |   | 4.0     | inspire.ec.europa.eu/schemas/lcv              | inspire.ec.europa.eu/schemas/lcv/4.0/LandCoverVector.xsd                                              
lu               |       |   | 4.0     | inspire.ec.europa.eu/schemas/lunom            | inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd                                        
lunom            |       |   | 4.0     | inspire.ec.europa.eu/schemas/lunom            | inspire.ec.europa.eu/schemas/lunom/4.0/LandUseNomenclature.xsd                                        
mr-core          |       |   | 4.0     | inspire.ec.europa.eu/schemas/mr-core          | inspire.ec.europa.eu/schemas/mr-core/4.0/MineralResourcesCore.xsd                                     
mu               |       |   |         | inspire.ec.europa.eu/schemas/mu/3.0rc3        | inspire.ec.europa.eu/schemas/mu/3.0rc3/MaritimeUnits.xsd                                              
net              |       |   | 4.0     | inspire.ec.europa.eu/schemas/net              | inspire.ec.europa.eu/schemas/net/4.0/Network.xsd                                                      
nz-core          |       |   | 4.0     | inspire.ec.europa.eu/schemas/nz-core          | inspire.ec.europa.eu/schemas/nz-core/4.0/NaturalRiskZonesCore.xsd                                     
nz               |       |   | 0.0     | inspire.ec.europa.eu/schemas/nz               | inspire.ec.europa.eu/schemas/nz/0.0/NaturalRiskZones.xsd                                              
of               |       |   | 4.0     | inspire.ec.europa.eu/schemas/of               | inspire.ec.europa.eu/schemas/of/4.0/OceanFeatures.xsd                                                 
oi               |       |   | 4.0     | inspire.ec.europa.eu/schemas/oi               | inspire.ec.europa.eu/schemas/oi/4.0/Orthoimagery.xsd                                                  
omop             |       |   | 3.0     | inspire.ec.europa.eu/schemas/omop             | inspire.ec.europa.eu/schemas/omop/3.0/ObservableProperties.xsd                                        
omor             |       |   | 3.0     | inspire.ec.europa.eu/schemas/omor             | inspire.ec.europa.eu/schemas/omor/3.0/ObservationReferences.xsd                                       
ompr             |       |   | 3.0     | inspire.ec.europa.eu/schemas/ompr             | inspire.ec.europa.eu/schemas/ompr/3.0/Processes.xsd                                                   
omso             |       |   | 3.0     | inspire.ec.europa.eu/schemas/omso             | inspire.ec.europa.eu/schemas/omso/3.0/SpecialisedObservations.xsd                                     
pd               |       |   | 4.0     | inspire.ec.europa.eu/schemas/pd               | inspire.ec.europa.eu/schemas/pd/4.0/PopulationDistributionDemography.xsd                              
pf               |       |   | 4.0     | inspire.ec.europa.eu/schemas/pf               | inspire.ec.europa.eu/schemas/pf/4.0/ProductionAndIndustrialFacilities.xsd                             
plu              |       |   | 4.0     | inspire.ec.europa.eu/schemas/plu              | inspire.ec.europa.eu/schemas/plu/4.0/PlannedLandUse.xsd                                               
ps               |       |   | 4.0     | inspire.ec.europa.eu/schemas/ps               | inspire.ec.europa.eu/schemas/ps/4.0/ProtectedSites.xsd                                                
sd               |       |   | 4.0     | inspire.ec.europa.eu/schemas/sd               | inspire.ec.europa.eu/schemas/sd/4.0/SpeciesDistribution.xsd                                           
selu             |       |   | 4.0     | inspire.ec.europa.eu/schemas/selu             | inspire.ec.europa.eu/schemas/selu/4.0/SampledExistingLandUse.xsd                                      
so               |       |   | 4.0     | inspire.ec.europa.eu/schemas/so               | inspire.ec.europa.eu/schemas/so/4.0/Soil.xsd                                                          
sr               |       |   | 4.0     | inspire.ec.europa.eu/schemas/sr               | inspire.ec.europa.eu/schemas/sr/4.0/SeaRegions.xsd                                                    
su-core          |       |   | 4.0     | inspire.ec.europa.eu/schemas/su-core          | inspire.ec.europa.eu/schemas/su-core/4.0/StatisticalUnitCore.xsd                                      
su-grid          |       |   | 4.0     | inspire.ec.europa.eu/schemas/su-grid          | inspire.ec.europa.eu/schemas/su-grid/4.0/StatisticalUnitGrid.xsd                                      
su-vector        |       |   | 4.0     | inspire.ec.europa.eu/schemas/su-vector        | inspire.ec.europa.eu/schemas/su-vector/4.0/StatisticalUnitVector.xsd                                  
su               |       |   | 0.0     | inspire.ec.europa.eu/schemas/su               | inspire.ec.europa.eu/schemas/su/0.0/StatisticalUnits.xsd                                              
tn-a             |       |   | 4.0     | inspire.ec.europa.eu/schemas/tn-a             | inspire.ec.europa.eu/schemas/tn-a/4.0/AirTransportNetwork.xsd                                         
tn-c             |       |   | 4.0     | inspire.ec.europa.eu/schemas/tn-c             | inspire.ec.europa.eu/schemas/tn-c/4.0/CableTransportNetwork.xsd                                       
tn-ra            |       |   | 4.0     | inspire.ec.europa.eu/schemas/tn-ra            | inspire.ec.europa.eu/schemas/tn-ra/4.0/RailwayTransportNetwork.xsd                                    
tn-ro            |       |   | 4.0     | inspire.ec.europa.eu/schemas/tn-ro            | inspire.ec.europa.eu/schemas/tn-ro/4.0/RoadTransportNetwork.xsd                                       
tn-w             |       |   | 4.0     | inspire.ec.europa.eu/schemas/tn-w             | inspire.ec.europa.eu/schemas/tn-w/4.0/WaterTransportNetwork.xsd                                       
tn               |       |   | 4.0     | inspire.ec.europa.eu/schemas/tn               | inspire.ec.europa.eu/schemas/tn/4.0/CommonTransportElements.xsd                                       
ugs              |       |   | 0.0     | inspire.ec.europa.eu/schemas/ugs              | inspire.ec.europa.eu/schemas/ugs/0.0/UtilityAndGovernmentalServices.xsd                               
us-emf           |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-emf           | inspire.ec.europa.eu/schemas/us-emf/4.0/EnvironmentalManagementFacilities.xsd                         
us-govserv       |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-govserv       | inspire.ec.europa.eu/schemas/us-govserv/4.0/GovernmentalServices.xsd                                  
us-net-common    |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-net-common    | inspire.ec.europa.eu/schemas/us-net-common/4.0/UtilityNetworksCommon.xsd                              
us-net-el        |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-net-el        | inspire.ec.europa.eu/schemas/us-net-el/4.0/ElectricityNetwork.xsd                                     
us-net-ogc       |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-net-ogc       | inspire.ec.europa.eu/schemas/us-net-ogc/4.0/OilGasChemicalsNetwork.xsd                                
us-net-sw        |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-net-sw        | inspire.ec.europa.eu/schemas/us-net-sw/4.0/SewerNetwork.xsd                                           
us-net-tc        |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-net-tc        | inspire.ec.europa.eu/schemas/us-net-tc/4.0/TelecommunicationsNetwork.xsd                              
us-net-th        |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-net-th        | inspire.ec.europa.eu/schemas/us-net-th/4.0/ThermalNetwork.xsd                                         
us-net-wa        |       |   | 4.0     | inspire.ec.europa.eu/schemas/us-net-wa        | inspire.ec.europa.eu/schemas/us-net-wa/4.0/WaterNetwork.xsd                                           
wfd              |       |   | 0.0     | inspire.ec.europa.eu/schemas/wfd              | inspire.ec.europa.eu/schemas/wfd/0.0/WaterFrameworkDirective.xsd                                      

"""

##

_XMLNS = 'xmlns'
_XSI = 'xsi'
_XSI_URL = 'www.w3.org/2001/XMLSchema-instance'

# fake namespace for 'xmlns:'
_ALL.append(gws.XmlNamespace(uid=_XMLNS, xmlns=_XMLNS, uri='', schemaLocation='', version='', isDefault=True))


def _load_known():
    def http(u):
        return 'http://' + u if not u.startswith('http') else u

    for ln in _KNOWN_NAMESPACES.strip().split('\n'):
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        uid, xmlns, dflt, version, uri, schema = [s.strip() for s in ln.split('|')]
        _ALL.append(gws.XmlNamespace(
            uid=uid,
            xmlns=xmlns or uid,
            uri=http(uri),
            schemaLocation=http(schema) if schema else '',
            version=version,
            isDefault=dflt != 'N'
        ))


_load_known()
_build_index()
