"""CSW DescribeRecord template (ISO).

Uses "MDRecordSchemaComponent.xml" from:

    GeoServer "csw-iso" plugin
    Licensed under the GNU General Public License
    © Copyright 2024, Open Source Geospatial Foundation
"""

import gws
import gws.base.ows.server as server
import gws.lib.mime


def main(ta: server.TemplateArgs):
    schema = gws.u.read_file(gws.c.APP_DIR + '/gws/plugin/ows_server/csw/templates/MDRecordSchemaComponent.xml')

    xml = f'''
        <?xml version="1.0" encoding="UTF-8"?>
		<csw:DescribeRecordResponse 
		    version="2.0.2"
			xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" 
			xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
			xsi:schemaLocation="http://www.opengis.net/cat/csw/2.0.2 http://schemas.opengis.net/csw/2.0.2/CSW-discovery.xsd">
		{schema}
		</csw:DescribeRecordResponse>
	'''
    return gws.ContentResponse(mime=gws.lib.mime.XML, content=xml.strip())
