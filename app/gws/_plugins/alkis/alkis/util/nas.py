import re
import zipfile

from bs4 import BeautifulSoup


def _as_dict(node):
    d = {'tag': node.name}

    for sub in node:
        if sub.name == 'dictionaryEntry':
            continue
        if sub.name in ('supertypeRef', 'valueTypeRef'):
            # for ref tags the value is their (only) href attr
            d[sub.name] = sub['xlink:href']
        elif sub.string.strip():
            d[sub.name] = str(sub.string)

    for k, v in d.items():
        m = re.match(r'urn:x-ogc:def:.+?:GeoInfoDok::adv:[^:]+:(.+)', v)
        if m:
            d[k] = m.group(1).lower()

    return d


def _parse(xml):
    bs = BeautifulSoup(xml, 'lxml-xml')

    """
        Examples of tags:

        Normal prop:

        <PropertyDefinition gml:id="S.084.0120.07.783">
          <identifier>urn:x-ogc:def:propertyType:GeoInfoDok::adv:6.0:AX_Buchungsstelle:beschreibungDesSondereigentums</identifier>
          <name>beschreibungDesSondereigentums</name>
          <cardinality>0..1</cardinality>
          <valueTypeName>CharacterString</valueTypeName>
          <type>attribute</type>
        </PropertyDefinition>

        Reference:

        <PropertyDefinition gml:id="G.344">
            <identifier>urn:x-ogc:def:propertyType:GeoInfoDok::adv:6.0:AX_Flurstueck:istGebucht</identifier>
            <name>istGebucht</name>
            <cardinality>1</cardinality>
            <valueTypeRef xlink:href="urn:x-ogc:def:featureType:GeoInfoDok::adv:6.0:AX_Buchungsstelle"/>
            <type>associationRole</type>
        </PropertyDefinition>

        ListedValue:

        <ListedValueDefinition gml:id="S.084.0120.08.528_S.084.0120.08.532">
          <identifier>urn:x-ogc:def:propertyType:GeoInfoDok::adv:6.0:AX_Wald:vegetationsmerkmal:1100</identifier>
          <name>Laubholz</name>
        </ListedValueDefinition>

    """

    for node in bs.find_all(['PropertyDefinition', 'ListedValueDefinition', 'TypeDefinition']):
        yield _as_dict(node)


def _extract(path_to_nas_zip):
    elems = []
    zf = zipfile.ZipFile(path_to_nas_zip)
    for fi in zf.infolist():
        # we only need Axxx.definitions.xml from the definitions folder
        if re.search(r'/definitions/A.+?.definitions.xml', fi.filename):
            with zf.open(fi) as fp:
                for elem in _parse(fp.read()):
                    elems.append(elem)

    return elems


def parse_properties(path_to_nas_zip):
    return {
        e['identifier']: e['name']
        for e in _extract(path_to_nas_zip)
        if e['tag'] == 'ListedValueDefinition'
    }
