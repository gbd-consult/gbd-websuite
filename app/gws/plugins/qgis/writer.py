"""Manipulate QGIS project files."""

import bs4

import gws


def add_variables(source_text: str, d: dict) -> str:
    """Inject our variables into a project"""

    # @TODO rewrite relative paths to absolute
    bs = bs4.BeautifulSoup(source_text, 'lxml-xml')

    """
    The vars are stored like this in both 2 and 3:
    
    <qgis>
    ....
        <properties>
            ....
            <Variables>
              <variableNames type="QStringList">
                <value>ONE</value>
                <value>TWO</value>
              </variableNames>
              <variableValues type="QStringList">
                <value>11</value>
                <value>22</value>
              </variableValues>
            </Variables>
        </properties>
    </qgis>
    
    """

    props = bs.properties
    if not props:
        props = bs.new_tag('properties')
        bs.append(props)

    if props.Variables:
        vs = dict(zip(
            [str(v.string) for v in props.select('Variables variableNames value')],
            [str(v.string) for v in props.select('Variables variableValues value')],
        ))
        props.Variables.decompose()
    else:
        vs = {}

    vs.update(d)

    props.append(bs.new_tag('Variables'))
    vnames = bs.new_tag('variableNames', type='QStringList')
    vvals = bs.new_tag('variableValues', type='QStringList')

    props.Variables.append(vnames)
    props.Variables.append(vvals)

    for k, v in sorted(vs.items()):
        v = gws.as_str(v).replace('\n', ' ').strip()
        if v:
            tag = bs.new_tag('value')
            tag.append(k)
            vnames.append(tag)

            tag = bs.new_tag('value')
            tag.append(v)
            vvals.append(tag)

    return str(bs)


