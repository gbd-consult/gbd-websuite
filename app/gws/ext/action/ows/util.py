import gws.tools.xml3
import gws.types as t
import gws.common.project
import gws.web.auth

tag = gws.tools.xml3.tag


def xml_response(tag, status=200):
    return t.HttpResponse({
        'mimeType': 'text/xml',
        'content': gws.tools.xml3.string(tag, compress=True),
        'status': status,
    })


def xml_exception_response(version, status, description):
    return xml_response(
        tag('ServiceExceptionReport',
            tag('ServiceException', description, {'code': status}),
            {
                'version': version,
                'xmlns': 'http://www.opengis.net/ogc',
                'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                'xsi:schemaLocation': 'http://www.opengis.net/ogc http://schemas.opengis.net/wms/1.3.0/exceptions_1_3_0.xsd',
            }),
        status=status
    )
